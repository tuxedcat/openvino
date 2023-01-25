# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from utils import utils
from argparse import ArgumentParser
from subprocess import Popen, STDOUT, TimeoutExpired
from hashlib import sha256
from pathlib import Path
from shutil import rmtree, copyfile
from zipfile import ZipFile, is_zipfile

import os
import sys
import threading
import platform
import csv
import datetime
import tarfile

if sys.version_info.major >= 3:
    import _thread as thread
else:
    import thread

FILENAME_LENGTH = 255
LOG_NAME_REPLACE_STR = "##NAME##"
DEFAULT_PROCESS_TIMEOUT = 3600
MAX_LENGHT = 4096 if platform.system() != "Windows" else 8191
TEST_STATUS = {
    'passed': ["[       OK ]"],
    'failed': ["[  FAILED  ]"],
    'hanged': ["Test finished by timeout"],
    'crashed': ["Unexpected application crash with code", "Segmentation fault", "Crash happens", "core dumped"],
    'skipped': ["[  SKIPPED ]"],
    'interapted': ["interapted", "Killed"]}
RUN = "[ RUN      ]"
GTEST_FILTER = "Google Test filter = "
DISABLED_PREFIX = "DISABLED_"

logger = utils.get_logger('test_parallel_runner')

def parse_arguments():
    parser = ArgumentParser()
    exec_file_path_help = "Path to the test executable file"
    cache_path_help = "Path to the cache file with test_name list sorted by execution time. .lst file!"
    worker_num_help = "Worker number. Default value is `cpu_count-1` "
    working_dir_num_help = "Working dir"
    process_timeout_help = "Process timeout in s"

    parser.add_argument("-e", "--exec_file", help=exec_file_path_help, type=str, required=True)
    parser.add_argument("-c", "--cache_path", help=cache_path_help, type=str, required=False, default="")
    parser.add_argument("-j", "--workers", help=worker_num_help, type=int, required=False, default=(os.cpu_count() - 1) if os.cpu_count() > 2 else 1)
    parser.add_argument("-w", "--working_dir", help=working_dir_num_help, type=str, required=False, default=".")
    parser.add_argument("-t", "--process_timeout", help=process_timeout_help, type=int, required=False, default=DEFAULT_PROCESS_TIMEOUT)
    return parser.parse_args()

def get_test_command_line_args():
    command_line_args = list()
    for i in range(len(sys.argv)):
        if sys.argv[i] == '--':
            command_line_args = sys.argv[i+1:]
            sys.argv = sys.argv[:i]
            break
    return command_line_args

# Class to read test cache    
class TestStructure:
    _name = ""
    _time = 0

    def __init__(self, name, time):
        self._name = name
        self._time = int(time)

class TaskManager:
    process_timeout = -1

    def __init__(self, command_list:list, working_dir: os.path):
        self._command_list = command_list
        self._process_list = list()
        self._workers = list()
        self._timers = list()
        self._idx = 0
        self._log_filename = os.path.join(working_dir, f"log_{LOG_NAME_REPLACE_STR}.log")

    def __create_thread(self, func):
        thread = threading.Thread(target=func)
        thread.daemon = True
        thread.start()
        return thread

    def init_worker(self):
        if len(self._command_list) <= self._idx:
            logger.warning(f"Skip worker initialiazation. Command list lenght <= worker index")
            return
        log_file_name = self._log_filename.replace(LOG_NAME_REPLACE_STR, str(self._idx))
        with open(log_file_name, "w") as log_file:
            worker = self.__create_thread(
                self._process_list.append(Popen(self._command_list[self._idx], shell=True, stdout=log_file, stderr=log_file)))
            self._workers.append(worker)
            worker.join()
            self._timers.append(datetime.datetime.now())
            log_file.close()
        # logger.info(f"{self._idx}/{len(self._command_list)} is started")
        self._idx += 1
    
    def __find_free_process(self):
        while True:
            for pid in range(len(self._process_list)):
                try:
                    if float((datetime.datetime.now() - self._timers[pid]).total_seconds()) > self.process_timeout:
                        logger.warning(f"Process {pid} exceed time limetattion per process")
                        self._process_list[pid].kill()
                    self._process_list[pid].wait(timeout=0)
                    # logger.info(f"{self._idx}/{len(self._command_list)} is started")
                    return pid
                except TimeoutExpired:
                    continue

    def __update_process(self, pid:int, log_file):
        self._process_list[pid] = Popen(self._command_list[self._idx], shell=True, stdout=log_file, stderr=log_file)

    def update_worker(self):
        if self._idx >= len(self._command_list):
            return False
        pid = self.__find_free_process()
        log_file_name = self._log_filename.replace(LOG_NAME_REPLACE_STR, str(self._idx))
        with open(log_file_name, "w") as log_file:
            self._workers[pid] = self.__create_thread(self.__update_process(pid, log_file))
            self._workers[pid].join()
            self._timers[pid] = datetime.datetime.now()
        self._idx += 1
        return True
    
    def compelete_all_processes(self):        
        while len(self._process_list) > 0:
            for pid in range(len(self._process_list)):
                try:
                    if float((datetime.datetime.now() - self._timers[pid]).total_seconds()) > self.process_timeout:
                        logger.warning(f"Process {pid} exceed time limetation per process. The process will be killed")
                        self._process_list[pid].kill()
                    self._process_list[pid].wait(timeout=0)
                    # logger.info(f"Process {pid} takes {float((datetime.datetime.now() - self._timers[pid]).total_seconds())}")
                    self._process_list.pop(pid)
                    logger.info(f"Compeleting processes: Active process counter: {len(self._process_list)}...")
                    break
                except TimeoutExpired:
                    continue

class TestParallelRunner:
    def __init__(self, exec_file_path: os.path, test_command_line: list, worker_num: int, working_dir: os.path, cache_path: os.path):
        self._exec_file_path = exec_file_path
        self._working_dir = working_dir
        self._command = self.__init_basic_command_line_for_exec_file(test_command_line)
        self._worker_num = worker_num
        if not os.path.exists(self._working_dir):
            os.mkdir(self._working_dir)
        if cache_path == "":
            cache_path = os.path.join(self._working_dir, "test_cache.lst")
        self._cache_path = os.path.join(cache_path)
        head, _ = os.path.split(self._cache_path)
        if not os.path.exists(head):
            os.mkdir(head)
        self._is_save_cache = True
        self._disabled_tests = list()
        self._total_test_cnt = 0

    def __unzip_archieve(self, zip_path: os.path):
        _, tail = os.path.split(zip_path)
        dst_path = os.path.join(self._working_dir, tail)
        copyfile(zip_path, dst_path)
        logger.info(f"Archieve {zip_path} was copied to {dst_path}")
        dst_dir, _ = os.path.splitext(dst_path)
        if tarfile.is_tarfile(zip_path):
            file = tarfile.open(dst_path)
            file.extractall(dst_dir)
            file.close()
        elif is_zipfile(zip_path):
            with ZipFile(dst_path, 'r') as zObject:
                zObject.extractall(path=dst_dir)
        else:
            logger.error(f"Impossible to extract {zip_path}")
            sys.exit(-1)
        logger.info(f"Archieve {dst_path} was extacted to {dst_dir}")
        os.remove(dst_path)
        logger.info(f"Archieve {dst_path} was removed")
        return dst_dir


    def __init_basic_command_line_for_exec_file(self, test_command_line: list):
        command = f'{self._exec_file_path}'
        is_input_folder = False
        for argument in test_command_line:
            if "--input_folders" in argument:
                is_input_folder = True
                command += f" --input_folders"
                argument = argument[argument.find("=")+1:]
            if is_input_folder and argument[0] != "-":
                buf = ""
                for _ in argument.split(','):
                    input_path = argument.replace('"', '')
                    if os.path.isfile(input_path) and (tarfile.is_tarfile(input_path) or is_zipfile(input_path)):
                        input_path = self.__unzip_archieve(input_path)
                    buf = utils.prepare_filelist(input_path, "*.xml", logger)
                    buf += ","
                argument = buf 
            else:
                is_input_folder = False
            command += f" {argument}"
        return command

    @staticmethod
    def __replace_restricted_symbols(input_string:str):
        restricted_symbols = "!@$%^&-+`~:;\",<>?"
        for symbol in restricted_symbols:
            input_string = input_string.replace(symbol, '*')
        return input_string


    def __get_test_list_by_runtime(self):
        test_list_file_name = os.path.join(self._working_dir, "test_list.lst")
        command_to_get_test_list = self._command + f' --gtest_list_tests >> {test_list_file_name}'
        logger.info(f"Get test list using command: {command_to_get_test_list}")
        Popen(command_to_get_test_list, shell=True).communicate()

        test_list = list()
        with open(test_list_file_name) as test_list_file:
            test_suite = ""
            for test_name in test_list_file.read().split('\n'):
                pos = test_name.find('#')
                if pos > 0:
                    real_test_name = test_suite + test_name[2:pos-2]
                    if DISABLED_PREFIX in real_test_name:
                        self._disabled_tests.append(real_test_name)
                    else:
                        test_list.append(f'"{self.__replace_restricted_symbols(real_test_name)}":')
                else:
                    test_suite = test_name
            test_list_file.close()
        os.remove(test_list_file_name)
        logger.info(f"Len test_list_runtime (without disabled tests): {len(test_list)}")
        return test_list


    def __get_test_list_by_cache(self):
        test_list_cache = list()
        if os.path.isfile(self._cache_path):
            logger.info(f"Get test list from cache file: {self._cache_path}")
            with open(self._cache_path, "r") as cache_file:
                for line in cache_file.readlines():
                    pos = line.find(":")
                    time = line[:pos]
                    test_name = line[pos+1:]
                    if not DISABLED_PREFIX in test_name:
                        test_list_cache.append(TestStructure(test_name.replace("\n", ""), time))
        logger.info(f"Len test_list_cache: {len(test_list_cache)}")
        return test_list_cache


    def __generate_proved_test_list(self, test_list: list, test_list_runtime:list):
        proved_test_list = list()
        if len(test_list) == len(test_list_runtime):
            proved_test_list = test_list
        else:
            for test in test_list:
                if test._name in test_list_runtime:
                    proved_test_list.append(test)

        if len(proved_test_list) < len(test_list_runtime):
            logger.warning(f'Cache file is not relevant the run. The test list will be taken from runtime')
            return []
        else:
            return proved_test_list


    def __prepare_smart_filters(self, proved_test_list:list):
        res_test_filters = list()
        def_length = len(self._command) + len(" --gtest_filter=")
        idx = len(proved_test_list)
        for i in range(len(proved_test_list)):
            if proved_test_list[i]._time == -1:
                idx = i
                break
        
        # Run crashed tests in a separed thread
        if idx < len(proved_test_list):
            while proved_test_list[idx]._time == -1:
                proved_test_list.pop(idx)

        # prepare gtest filters per worker according command line length limitation
        while len(proved_test_list) > 0:
            test_times = []
            is_not_full = True
            worker_test_filters = list()

            for _ in range(self._worker_num):
                if len(proved_test_list) == 0:
                    break
                worker_test_filters.append(f'"{self.__replace_restricted_symbols(proved_test_list[0]._name)}":')
                test_times.append(proved_test_list[0]._time)
                proved_test_list.pop(0)
            while is_not_full and len(proved_test_list) > 0:
                for i in range(self._worker_num):
                    if i >= len(proved_test_list):
                        break
                    if i == 0:
                        continue
                    while test_times[0] > test_times[i] + proved_test_list[len(proved_test_list) - 1]._time:
                        final_pos = len(proved_test_list) - 1
                        if len(worker_test_filters[i]) + def_length + len(proved_test_list[final_pos]._name) < MAX_LENGHT:
                            worker_test_filters[i] += f'"{self.__replace_restricted_symbols(proved_test_list[final_pos]._name)}":'
                            test_times[i] += proved_test_list[final_pos]._time
                            proved_test_list.pop(final_pos)
                        else:
                            is_not_full = False
                            break
                if is_not_full and len(proved_test_list) > 0:
                    worker_test_filters[0] += f'"{self.__replace_restricted_symbols(proved_test_list[0]._name)}":'
                    test_times[0] += proved_test_list[0]._time
                    proved_test_list.pop(0)
            for filter in worker_test_filters:
                res_test_filters.append(filter)
            is_not_full = True
        # logging for debug
        # for i in range(len(res_test_filters)):
        #     filter = res_test_filters[i]
        #     cnt = filter.count('\":')
        #     logger.info(f"Number of tests in job_{i}: {cnt}")
        return res_test_filters
            
    def __get_filters(self):
        if not os.path.isfile(self._exec_file_path):
            logger.error(f"Test executable file {self._exec_file_path} is not exist!")
            sys.exit(-1)

        test_list_runtime = self.__get_test_list_by_runtime()
        test_list_cache = self.__get_test_list_by_cache()
        proved_test_list = self.__generate_proved_test_list(test_list_cache, test_list_runtime)
        final_test_list = list()              
        if len(proved_test_list) > 0:
            self._is_save_cache = False
            logger.info(f"Test list is taken from cache.")
            self._total_test_cnt = len(proved_test_list)
            final_test_list = self.__prepare_smart_filters(proved_test_list)
        else:
            logger.info(f"Test list is taken from runtime.")
            self._total_test_cnt = len(test_list_runtime)
            final_test_list = test_list_runtime
            final_test_list.reverse()
        logger.info(f"Total test counter is {self._total_test_cnt}")
        return final_test_list

    @staticmethod
    def progressbar(it_num, message="", progress_bar_size=60, out=sys.stdout):
        max_len = len(it_num)
        def show(sym_pos):
            x = int(progress_bar_size * sym_pos / max_len)
            print("{}[{}{}] {}/{}".format(message, "#"*x, "."*(progress_bar_size-x), sym_pos, max_len), 
                    end='\r', file=out, flush=True)
        show(0)
        for i, item in enumerate(it_num):
            yield item
            show(i+1)
        print("", flush=True, file=out)

    def run(self):
        if TaskManager.process_timeout == -1:
            TaskManager.process_timeout = DEFAULT_PROCESS_TIMEOUT
        logger.info(f"Run test parallel is started. Worker num is {self._worker_num}")
        t_start = datetime.datetime.now()
        
        commands = [f'{self._command} --gtest_filter={filter}' for filter in self.__get_filters()]
        task_manager = TaskManager(commands, self._working_dir)
        # from tqdm import tqdm
        # for _ in tqdm(range(self._worker_num)):
        for _ in self.progressbar(range(self._worker_num), "Worker initialization: ", 40):
            task_manager.init_worker()
        for _ in self.progressbar(range(len(commands) - self._worker_num), "Worker execution: ", 40):
            if not task_manager.update_worker():
                break
        task_manager.compelete_all_processes()

        t_end = datetime.datetime.now()
        logger.info(f"Run test parallel is finished successfully. Total time is {(t_end - t_start).total_seconds()}s")


    def postprocess_logs(self):
        test_results = dict()
        logger.info(f"Log analize is started")
        def __save_log(logs_dir, dir, test_name):
            test_log_filename = os.path.join(logs_dir, dir, f"{test_name}.txt".replace('/', '_'))
            hash_str = str(sha256(test_name.encode('utf-8')).hexdigest())
            if hash_str in hash_map.keys():
                # logger.warning(f"Test {test_name} was executed before!")
                return False
            else:
                hash_map.update({hash_str: (dir, test_name)})
            test_log_filename = os.path.join(logs_dir, dir, f'{hash_str}.txt')
            if os.path.isfile(test_log_filename):
                # logger.warning(f"Log file {test_log_filename} is exist!")
                return False
            with open(test_log_filename, "w") as log:
                log.writelines(test_log)
                log.close()
            return True

        logs_dir = os.path.join(self._working_dir, "logs")
        if os.path.exists(logs_dir):
            logger.info(f"Logs directory {logs_dir} is cleaned up")
            rmtree(logs_dir)
        os.mkdir(logs_dir)
        for test_st, _ in TEST_STATUS.items():
            if not os.path.exists(os.path.join(logs_dir, test_st)):
                os.mkdir(os.path.join(logs_dir, test_st))
        hash_map = dict()
        test_times = list()
        for log in Path(self._working_dir).rglob("log_*.log"):
            log_filename = os.path.join(self._working_dir, log)
            with open(log_filename, "r") as log_file:
                test_name = None
                test_log = list()
                dir = None
                test_cnt_expected = test_cnt_real_saved_now = test_cnt_real_saved_before = 0
                for line in log_file.readlines():
                    if GTEST_FILTER in line:
                        line = line[line.find(GTEST_FILTER):]
                        test_cnt_expected = line.count(':')
                    if RUN in line:
                        test_name = line[line.find(RUN) + len(RUN) + 1:-1:]
                    if dir is None:
                        for test_st, mes_list in TEST_STATUS.items():
                            for mes in mes_list:
                                if mes in line:
                                    dir = test_st
                                    break
                            if not dir is None:
                                break
                    if test_name is not None:
                        test_log.append(line)
                        if dir:
                            if __save_log(logs_dir, dir, test_name):
                                # update test_cache with tests. If tests is crashed use -2 as unknown time
                                time = -1
                                if "ms)" in line:
                                    time = line[line.rfind("(") + 1:line.rfind("ms)") - 1]
                                test_times.append((int(time), test_name))
                                if dir in test_results.keys():
                                    test_results[dir] += 1
                                else:
                                    test_results[dir] = 1
                                test_cnt_real_saved_now += 1
                                test_name = None
                                test_log = list()
                                dir = None
                            else:
                                test_cnt_real_saved_before += 1
                log_file.close()
                test_cnt_real = test_cnt_real_saved_before + test_cnt_real_saved_now
                if test_cnt_real < test_cnt_expected:
                    logger.error(f"Number of tests in {log}: {test_cnt_real}. Expected is {test_cnt_expected} tests")
                else:
                    os.remove(log_filename)
        if self._is_save_cache:
            test_times.sort(reverse=True)
            with open(self._cache_path, "w") as cache_file:
                cache_file.writelines([f"{time}:\"" + test_name + "\":\n" for time, test_name in test_times])
                cache_file.close()
                logger.info(f"Test cache test is saved to: {self._cache_path}")
        hash_table_path = os.path.join(logs_dir, "hash_table.csv")
        with open(hash_table_path, "w") as csv_file:
            csv_writer = csv.writer(csv_file, dialect='excel')
            csv_writer.writerow(["Dir", "Hash", "Test Name"])
            for hash, st in hash_map.items():
                dir, name = st
                csv_writer.writerow([dir, hash, name])
            logger.info(f"Hashed test list is saved to: {hash_table_path}")


        disabled_tests_path = os.path.join(logs_dir, "disabled_tests.lst")
        with open(disabled_tests_path, "w") as disabled_tests_file:
            for i in range(len(self._disabled_tests)):
                self._disabled_tests[i] += "\n"
            disabled_tests_file.writelines(self._disabled_tests)
            disabled_tests_file.close()
            logger.info(f"Disabled test list is saved to: {disabled_tests_path}")

        is_successfull_run = True
        test_cnt = 0
        for test_st, test_res in test_results.items():
            logger.info(f"{test_st} test counter is: {test_res}")
            test_cnt += test_res
            if (test_st != "passed" and test_st != "skipped") and test_res > 0:
                is_successfull_run = False
        if len(self._disabled_tests):
            logger.info(f"disabled test counter is: {len(self._disabled_tests)}")
        if self._total_test_cnt != test_cnt:
            logger.error(f"Total test count is {test_cnt} is different with expected {self._total_test_cnt} tests")
            is_successfull_run = False
        logger.info(f"Total test count with disabled tests is {test_cnt + len(self._disabled_tests)}. All logs is saved to {logs_dir}")
        return is_successfull_run

if __name__ == "__main__":
    exec_file_args = get_test_command_line_args()
    args = parse_arguments()
    logger.info(f"[ARGUMENTS] --exec_file={args.exec_file}")
    logger.info(f"[ARGUMENTS] --working_dir={args.working_dir}")
    logger.info(f"[ARGUMENTS] --process_timeout={args.process_timeout}")
    logger.info(f"[ARGUMENTS] --cache_path={args.cache_path}")
    logger.info(f"[ARGUMENTS] Executable file arguments = {exec_file_args}")
    TaskManager.process_timeout = args.process_timeout
    conformance = TestParallelRunner(args.exec_file, exec_file_args, args.workers, args.working_dir, args.cache_path)
    conformance.run()
    if not conformance.postprocess_logs():
        logger.error("Run is not successful")
        sys.exit(-1)
    else:
        logger.info("Run is successful")