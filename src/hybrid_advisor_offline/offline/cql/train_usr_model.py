# 训练一个机器学习模型（logistic回归），用来预测客户是否会接受银行的定期存款推销。
import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report

# 定义文件路径
DATA_FILE = "./data/bm_full.csv"  # 输入数据文件
MODEL_DIR = "./models"  # 模型保存目录
MODEL_PATH = os.path.join(MODEL_DIR, "user_model.pkl")  # 最终模型文件路径

def get_model_features():
    """
    定义并返回用户接受度模型所使用的数值feature和类别feature
    """
    num_feature = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']
    categori_feature = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'day_of_week', 'month', 'poutcome']
    return num_feature, categori_feature # 返回一个tuple

def tarin_usr_accetp_model():
    """
    训练用户接受度模型，使用logistic 回归
    """
    print("-----开始训练用户接受度模型-----")

    # 1. --------加载数据
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"数据文件未找到，请先generate数据")
    df = pd.read_csv(DATA_FILE, sep=';')
    print(f"已经加载{len(df)}行数据")

    # 2. --------数据预处理
    # 清理原始数据集中列名可能包含的引号或空格
    df.columns = [col.strip().replace('"', '') for col in df.columns]

    # 目标变量是 'y'，表示客户是否接受了定期存款，接受1，不接受0
    X = df.drop('y', axis=1)
    y = df['y'].apply(lambda x: 1 if x.strip().replace('"', '') == 'yes' else 0)

    # 获取特征列表
    num_feature, categori_feature = get_model_features()

    # 划分训练集和测试集，stratify=y 实现分层抽样，确保拆分后训练集与测试集中目标变量 y 的类别比例与原始数据集保持一致。
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 为数值和类别特征创建预处理管道
    num_transformer = StandardScaler()  # 指定具体的处理工具，数值特征标准化
    categori_transformer = OneHotEncoder(handle_unknown='ignore')  # 类别特征独热编码

    # 使用 ColumnTransformer 创建一个统一的预处理器
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_feature),
            ('cat', categori_transformer, categori_feature)
        ],
        remainder='passthrough'  # 保留未指定的列（如果有的话）
    )

    # 创建包含预处理器和分类器的完整 Pipeline
    model_pipline = Pipeline(steps=[
        ('preprocessor',preprocessor),
        ('classifire',LogisticRegression(class_weight='balanced', max_iter=1000,random_state=42))
    ])

    # 3. ---------训练模型
    # 完成后，model_pipeline 就包含了已经用训练数据校准好的预处理器和已经训练好的逻辑回归模型
    print("正在训练模型 Pipeline...")
    model_pipline.fit(X_train, y_train)

    # 4. ---------评估模型
    print("\n-----------在测试集上评估模型性能:----------")
    y_pred = model_pipline.predict(X_test)
    print(f"准确率: {accuracy_score(y_test, y_pred):.4f}") #(预测正确的数量) / (总预测数量)
    print("\n分类报告:")
    print(classification_report(y_test, y_pred, target_names=['未接受', '接受']))

    # 保存训练好的模型 Pipeline
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        print(f"已创建目录: {MODEL_DIR}")
        
    joblib.dump(model_pipline,MODEL_PATH)
    # 模型预测的准确性严重依赖于新数据的预处理方式必须和训练数据完全一致
    print(f"\n模型 Pipeline 已保存至: {MODEL_PATH}")
