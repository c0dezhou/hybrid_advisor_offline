import os
from hybrid_advisor_offline.offline.cql.gen_datasets import DATA_DIR,download_mkt_data,download_user_data

def test_gen_datasets():
    """构造数据集"""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"创建文件夹: {DATA_DIR}")
    download_mkt_data()
    print("+"*30)
    download_user_data()
    print("\n数据集构造完毕")
