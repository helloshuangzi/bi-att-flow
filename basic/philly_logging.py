import logging
import os
import platform
import socket
import sys
import inspect

import tensorflow as tf


def get_os_name():
    return platform.system()


def get_host_name():
    return platform.node()


def get_host_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    return s.getsockname()[0]


def get_python_version_info():
    return sys.version_info


def get_environment_variable(name, default=None):
    if name not in os.environ:
        return default
    return os.environ[name]

def dump_philly_log(app_message):
    logging.basicConfig(stream=sys.stdout, format='[%(asctime)s] %(levelname)s: %(message)s', level=logging.INFO)

    logging.info("===============================")
    logging.info("os=%s", get_os_name())
    logging.info("host=%s", get_host_name())
    logging.info("ip=%s", get_host_ip())
    logging.info("python=%s", get_python_version_info()[0:3])
    logging.info("cwd=%s", os.getcwd())

    logging.info("===============================")
    logging.info("tf version=%s", tf.__version__)
    logging.info("os.environ=%s", os.environ)
    logging.info("sys.argv=%s", sys.argv)

    logging.info("===============================")
    logging.info("PYTHONPATH=%s", get_environment_variable("PYTHONPATH"))
    logging.info("sys.path=%s", sys.path)

    logging.info("===============================")
    logging.info(__file__)

    logging.info(app_message)

