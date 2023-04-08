# 导入需要的库
import streamlit as st
import pandas as pd
import joblib

st.header("基于机器学习预测首诊肺癌患者发生第二原发癌风险的研究")
st.sidebar.header('临床特征')

# Dropdown input 1
Age60= st.sidebar.selectbox("年龄", ("<60", ">=60"))
# Dropdown input 2
Race = st.sidebar.selectbox("性别", ("白种人", "黑种人", "亚裔", "美洲印第安人", "其他"))
# Dropdown input 3
Marital = st.sidebar.selectbox("婚姻状况", ("已婚", "离异", "单身", "其他"))
# Dropdown input 4
Histology = st.sidebar.selectbox("组织学类型", ("小细胞肺癌", "肺鳞癌", "肺腺癌", "其他"))
# Dropdown input 5
PrimarySite = st.sidebar.selectbox("肺癌原发部位", (
    "主支气管", "肺上叶", "肺中叶", "肺下叶", "其他"))
# Dropdown input 6
Laterality = st.sidebar.selectbox("偏侧性", ("左侧", "右侧", "其他/不详"))
# Dropdown input 7
Tumorsize = st.sidebar.selectbox("T分期", ("T0", "T1", "T2", "T3", "T4", "不详"))
# Dropdown input 8
Surgery = st.sidebar.selectbox("手术情况", ("否/不详", "是"))
# Dropdown input 9
LNpositive = st.sidebar.selectbox("淋巴结检出情况", ("阴性", "阳性", "不详"))





# 如果按下按钮
if st.button("预测"):  # 显示按钮
    # 加载训练好的模型
    model = joblib.load("xgb.pkl")
    # 将输入存储DataFrame
    X = pd.DataFrame([[Age60, Race, Marital, Histology, PrimarySite, Laterality, Tumorsize,Surgery,LNpositive]],
                     columns=['Age60', 'Race', 'Marital', 'Histology', 'PrimarySite', 'Laterality', 'Tumorsize', 'Surgery', 'LNpositive'])

    X['Age60'] = X['Age60'].replace(["<60", ">=60"], [1, 2])
    X['Race'] = X['Race'].replace(["白种人", "黑种人", "亚裔", "美洲印第安人", "其他"], [1, 2, 3, 4, 5])
    X['Marital'] = X['Marital'].replace(["已婚", "离异", "单身", "其他"], [1, 2, 3, 4])
    X['Histology'] = X['Histology'].replace(["小细胞肺癌", "肺鳞癌", "肺腺癌", "其他"], [1, 2, 3, 4])
    X['PrimarySite'] = X['PrimarySite'].replace(
        ["主支气管", "肺上叶", "肺中叶", "肺下叶", "其他"], [1, 2, 3, 4, 5])
    X['Laterality'] = X['Laterality'].replace(["左侧", "右侧", "其他/不详"], [1, 2, 3])
    X['Tumorsize'] = X['Tumorsize'].replace(["T0", "T1", "T2", "T3", "T4", "不详"], [0, 1, 2, 3, 4, 5])
    X['Surgery'] = X['Surgery'].replace(["否/不详", "是"], [0, 1])
    X['LNpositive'] = X['LNpositive'].replace(["阴性", "阳性", "不详"], [0, 1, 2])

    # 进行预测
    prediction = model.predict(X)[0]
    Predict_proba = model.predict_proba(X)[:, 1][0]
    # 输出预测结果
    if prediction == 0:
        st.subheader(f"SPC发生风险:  低风险")
    else:
        st.subheader(f"SPC发生风险:  高风险")
    # 输出概率
    st.subheader(f"SPC发生概率:  {'%.2f' % float(Predict_proba * 100) + '%'}")

