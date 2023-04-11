from causis_api.const import get_version
from causis_api.const import login

login.username = "yisen.du"
login.password = "Eason@20010917"
login.version = get_version()

from causis_api.data import *


def get_data():
    all_instrument_china = all_instruments(type='R', date='None', market='cn')
    print(all_instrument_china)


if __name__ == "__main__":
    get_data()
