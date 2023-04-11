from causis_api.const import get_version
from causis_api.const import login
import os

login.username = "yisen.du"
login.password = "Eason@20010917"
login.version = get_version()
from causis_api.data import *


def get_data():
    # download all the data I need for this project
    os.chdir("../../raw_data/")

    # 下载郑棉连续合约行情数据 编号：R.CN.CZC.CF.0004
    df_contract = get_price("R.CN.CZC.CF.0004",
                            "2005-02-21",
                            end_date=None,
                            frequency='day',
                            fields=None,
                            adjust_type=None,
                            skip_suspended=False,
                            market='cn',
                            expect_df=False)
    df_contract.to_csv("CF_price.csv")

    # 下载郑棉可交易主力合约
    df_dominant = get_dominant_contracts("R.CN.CZC.CF.0004",
                                         "2005-02-21",
                                         end_date=None)
    df_dominant.to_csv("CF_dominant.csv")

    # 下载基差相关指标：中国棉花价格指数:3128B，棉花最低交割价，棉花-基差因子
    df_spot = get_symbol_value('中国棉花价格指数:3128B', "2005-02-21")
    df_low = get_symbol_value('棉花最低交割价', "2005-02-21")
    df_basis_factor = get_symbol_value('棉花-基差因子', "2005-02-21")

    df_spot.to_csv("CF_basis_spot.csv")
    df_low.to_csv("CF_basis_lowPrice.csv")
    df_basis_factor.to_csv("CF_basis_factor.csv")

    # 下载利润相关指标：进口棉利润，棉花-利润因子
    df_profit = get_symbol_value('进口棉利润', "2005-02-21")
    df_profit_factor = get_symbol_value('棉花-利润因子', "2005-02-21")

    df_profit.to_csv("CF_profit.csv")
    df_profit_factor.to_csv("CF_profit_factor.csv")

    # 下载产量相关指标：全国棉花累计检验量
    df_supply = get_symbol_value('全国棉花累计检验量|DZ', "2005-02-21")

    df_supply.to_csv("CF_supply.csv")

    # 下载消费相关指标：棉花:抛储:成交量|DZ，轻纺城棉布日成交量|DZ
    df_consume1 = get_symbol_value('棉花:抛储:成交量|DZ', "2005-02-21")
    df_consume2 = get_symbol_value('轻纺城棉布日成交量|DZ', "2005-02-21")

    df_consume1.to_csv("CF_consume_1.csv")
    df_consume2.to_csv("CF_consume_2.csv")

    # 下载库存相关指标：纺企棉花库存，纺企棉纱库存，棉花-库存因子
    df_inventory_cotton = get_symbol_value('纺企棉花库存', "2005-02-21")
    df_inventory_yarn = get_symbol_value('纺企棉纱库存', "2005-02-21")
    df_inventory_factor = get_symbol_value('棉花-库存因子', "2005-02-21")

    df_inventory_cotton.to_csv("CF_inventory_cotton.csv")
    df_inventory_yarn.to_csv("CF_inventory_yarn.csv")
    df_inventory_factor.to_csv("CF_inventory_factor.csv")


if __name__ == "__main__":
    get_data()
