# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
# data operation package
import pandas as pd
import numpy as np
import streamlit as st

# show the result with picture
#import seaborn as sns
#from matplotlib import pyplot as plt
# show the datetime info
#from datetime import datetime
from PIL import Image

import warnings
warnings.filterwarnings("ignore")


def main():
    st.sidebar.subheader("数据展示选择:sunglasses:")
    select = st.sidebar.selectbox("请选择操作", ["初始数据", "数据认识", "特征热图", "预测结果"])
    st.sidebar.subheader(":flags:NOTE")
    st.sidebar.caption("初始数据：初始训练集和预测集的展示")
    st.sidebar.caption("数据认识：对初始数据集中相关特征的探索")
    st.sidebar.caption("特征热图：特征工程后对特征进行热图展示")
    st.sidebar.caption("预测结果：使用集成学习算法进行训练及预测")


    st.markdown("<h1 style= 'text-align: center; color: black;'>优惠券使用预测</h1>", unsafe_allow_html=True)
    img = Image.open("C:/课程/o2o/figs/1.png")
    st.image(img, width=820)

    @st.cache(allow_output_mutation=True)
    def load_pre_train():
        train_data = pd.read_csv('C:/课程/o2o/ccf_offline_stage1_train.csv', keep_default_na=False)
        return train_data

    @st.cache(allow_output_mutation=True)
    def load_pre_test():
        test_data = pd.read_csv('C:/课程/o2o/ccf_offline_stage1_test_revised.csv', keep_default_na=False)
        return test_data

    @st.cache(allow_output_mutation=True)
    def load_pre_online():
        online_data = pd.read_csv('C:/课程/o2o/ccf_online_stage1_train.csv', keep_default_na=False)
        return online_data

    @st.cache(allow_output_mutation=True)
    def load_rf():
        rf_data = pd.read_csv('C:/课程/o2o/result/rf_preds.csv', keep_default_na=False)
        return rf_data

    @st.cache(allow_output_mutation=True)
    def load_gbdt():
        gbdt_data = pd.read_csv('C:/课程/o2o/result/gbdt_preds.csv', keep_default_na=False)
        return gbdt_data

    @st.cache(allow_output_mutation=True)
    def load_xgb():
        xgb_data = pd.read_csv('C:/课程/o2o/result/xgb_preds.csv', keep_default_na=False)
        return xgb_data

    @st.cache
    def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
        return df.to_csv().encode('utf-8')

    pre_train = load_pre_train()
    pre_test = load_pre_test()
    pre_online = load_pre_online()
    train = pre_train
    test = pre_test
    ontrain = pre_online
    rf = load_rf()
    gbdt = load_gbdt()
    xgb = load_xgb()

    rf_csv=convert_df(rf)
    gbdt_csv=convert_df(gbdt)
    xgb_csv=convert_df(xgb)

    #if st.button('展示数据'):
    if select == "初始数据":
        data_select = st.selectbox("请选择加载的数据集", ["ccf_offline_stage1_train.csv", "ccf_online_stage1_train.csv", "ccf_offline_stage1_test_revised.csv"])
        if st.button('加载数据'):
            if data_select == 'ccf_offline_stage1_train.csv':
                st.subheader("线下训练集数据展示:book:")
                st.write(train.head(1000))
            elif data_select == 'ccf_online_stage1_train.csv':
                st.subheader("线上训练集数据展示:book:")
                st.write(ontrain.head(1000))
            else:
                st.subheader("测试集数据展示:book:")
                st.write(test.head(1000))
    elif select == "数据认识":
        data_select = st.selectbox("请选择加载的数据集", ["ccf_offline_stage1_train.csv", "ccf_online_stage1_train.csv",
                                                 "ccf_offline_stage1_test_revised.csv"])
        data_analyze = st.selectbox("请选择数据探索维度", ["用户活跃度", "商户活跃度", "优惠券概况", "距离分布(仅线下可选）"])
        if data_select == 'ccf_offline_stage1_train.csv':
            if data_analyze == "用户活跃度":
                if st.button('展示数据'):
                    st.subheader("用户活跃度展示:bar_chart:")
                    img = Image.open("C:/课程/o2o/figs/user_count.png")
                    st.image(img, width=820)
                    st.caption("0（流失用户）：仅领取过一次优惠券")
                    st.caption("1（低活跃度用户）：领取优惠券2-5次")
                    st.caption("2（中活跃度用户）：领取优惠券6-10次")
                    st.caption("3（高活跃度用户）：领取优惠券超过10次")
            elif data_analyze == "商户活跃度":
                if st.button('展示数据'):
                    st.subheader("商户活跃度展示:bar_chart:")
                    img = Image.open("C:/课程/o2o/figs/mer_count.png")
                    st.image(img, width=820)
                    st.caption("0：商家被领取优惠券次数低于20")
                    st.caption("1：商家被领取优惠券20-100次")
                    st.caption("2：商家被领取优惠券100-1000次")
                    st.caption("3：商家被领取优惠券超过1000次")
            elif data_analyze == "优惠券概况":
                if st.button('展示数据'):
                    st.subheader("优惠券活跃度展示:bar_chart:")
                    img = Image.open("C:/课程/o2o/figs/cou_count.png")
                    st.image(img, width=820)
                    st.caption("0：优惠券被领取次数低于10")
                    st.caption("1：优惠券被领取11-100次")
                    st.caption("2：优惠券被领取101-1000")
                    st.caption("3：优惠券被领取超过1000次")
                    st.subheader("优惠券折扣率分布:chart_with_upwards_trend:")
                    img = Image.open("C:/课程/o2o/figs/discount_rate.png")
                    st.image(img, width=820)
                    st.subheader("优惠券使用分布:chart_with_downwards_trend:")
                    img = Image.open("C:/课程/o2o/figs/coupon_use.png")
                    st.image(img, width=820)
                    st.subheader("优惠券领取周分布:bar_chart:")
                    img = Image.open("C:/课程/o2o/figs/cou_week.png")
                    st.image(img, width=820)
                    st.subheader("优惠券领取月分布:bar_chart:")
                    img = Image.open("C:/课程/o2o/figs/cou_month.png")
                    st.image(img, width=820)
            else:
                if st.button('展示数据'):
                    st.subheader("距离分布:bar_chart:")
                    img = Image.open("C:/课程/o2o/figs/distance from merchant.png")
                    st.image(img, width=820)
        elif data_select == 'ccf_online_stage1_train.csv':
            if data_analyze == "用户活跃度":
                if st.button('展示数据'):
                    st.subheader("用户活跃度展示:bar_chart:")
                    img = Image.open("C:/课程/o2o/figs/on_user_count.png")
                    st.image(img, width=820)
                    st.caption("0（流失用户）：仅领取过一次优惠券")
                    st.caption("1（低活跃度用户）：领取优惠券2-5次")
                    st.caption("2（中活跃度用户）：领取优惠券6-10次")
                    st.caption("3（高活跃度用户）：领取优惠券超过10次")
            elif data_analyze == "商户活跃度":
                if st.button('展示数据'):
                    st.subheader("商户活跃度展示:bar_chart:")
                    img = Image.open("C:/课程/o2o/figs/on_mer_count.png")
                    st.image(img, width=820)
                    st.caption("0：商家被领取优惠券次数低于20")
                    st.caption("1：商家被领取优惠券20-100次")
                    st.caption("2：商家被领取优惠券100-1000次")
                    st.caption("3：商家被领取优惠券超过1000次")
            elif data_analyze == "优惠券概况":
                if st.button('展示数据'):
                    st.subheader("优惠券活跃度展示:bar_chart:")
                    img = Image.open("C:/课程/o2o/figs/on_cou_count.png")
                    st.image(img, width=820)
                    st.caption("0：优惠券被领取次数低于10")
                    st.caption("1：优惠券被领取11-100次")
                    st.caption("2：优惠券被领取101-1000")
                    st.caption("3：优惠券被领取超过1000次")
                    st.subheader("优惠券折扣率分布:chart_with_upwards_trend:")
                    img = Image.open("C:/课程/o2o/figs/on_discount_rate.png")
                    st.image(img, width=820)
                    st.subheader("优惠券使用分布:chart_with_downwards_trend:")
                    img = Image.open("C:/课程/o2o/figs/on_coupon_use.png")
                    st.image(img, width=820)
            else:
                if st.button('展示数据'):
                    st.error("线上数据集没有距离字段:worried:")
                    st.snow()
        else:
            if data_analyze == "用户活跃度":
                if st.button('展示数据'):
                    st.subheader("用户活跃度展示:bar_chart:")
                    img = Image.open("C:/课程/o2o/figs/test_user_count.png")
                    st.image(img, width=820)
                    st.caption("0（流失用户）：仅领取过一次优惠券")
                    st.caption("1（低活跃度用户）：领取优惠券2-5次")
                    st.caption("2（中活跃度用户）：领取优惠券6-10次")
                    st.caption("3（高活跃度用户）：领取优惠券超过10次")
            elif data_analyze == "商户活跃度":
                if st.button('展示数据'):
                    st.subheader("商户活跃度展示:bar_chart:")
                    img = Image.open("C:/课程/o2o/figs/test_mer_count.png")
                    st.image(img, width=820)
                    st.caption("0：商家被领取优惠券次数低于20")
                    st.caption("1：商家被领取优惠券20-100次")
                    st.caption("2：商家被领取优惠券100-1000次")
                    st.caption("3：商家被领取优惠券超过1000次")
            elif data_analyze == "优惠券概况":
                if st.button('展示数据'):
                    st.subheader("优惠券活跃度展示:bar_chart:")
                    img = Image.open("C:/课程/o2o/figs/test_cou_count.png")
                    st.image(img, width=820)
                    st.caption("0：优惠券被领取次数低于10")
                    st.caption("1：优惠券被领取11-100次")
                    st.caption("2：优惠券被领取101-1000")
                    st.caption("3：优惠券被领取超过1000次")
                    st.subheader("优惠券折扣率分布:chart_with_upwards_trend:")
                    img = Image.open("C:/课程/o2o/figs/test_discount_rate.png")
                    st.image(img, width=820)
            else:
                if st.button('展示数据'):
                    st.subheader("距离分布:bar_chart:")
                    img = Image.open("C:/课程/o2o/figs/test distance from merchant.png")
                    st.image(img, width=820)


    elif select == "特征热图":
        heatmap = st.selectbox("请选择热图", ["全部特征热图", "上采样热图", "PCA及热图"])
        if heatmap == "全部特征热图":
            if st.button('展示数据'):
                st.subheader("全部特征热图:art:")
                img = Image.open("C:/课程/o2o/figs/heatmap_all.png")
                st.image(img, width=820)
        elif heatmap == "上采样热图":
            if st.button('展示数据'):
                st.subheader("上采样热图:art:")
                img = Image.open("C:/课程/o2o/figs/heatmap_after.png")
                st.image(img, width=820)
        else:
            if st.button('展示数据'):
                st.subheader("PCA:bar_chart:")
                img = Image.open("C:/课程/o2o/figs/PCA.png")
                st.image(img, width=820)
                st.subheader("热图:art:")
                img = Image.open("C:/课程/o2o/figs/heatmap_after.png")
                st.image(img, width=820)
    elif select == "预测结果":
        model = st.selectbox("请选择使用的模型", ["Random Forest", "GBDT", "XGBoost"])
        if model == "Random Forest":
            if st.button('展示数据'):
                st.subheader("K折交叉验证结果:wavy_dash:")
                img = Image.open("C:/课程/o2o/figs/RF_cross_val.png")
                st.image(img, width=820)
                st.subheader("示例验证（dataset1训练，dataset2预测）:bulb:")
                img = Image.open("C:/课程/o2o/figs/RF.png")
                st.image(img, width=820)
                st.subheader("预测（对dataset3进行预测）:mag_right:")
                st.dataframe(rf.head(1000))
                st.download_button(
                    label="Download data as CSV",
                    data=rf_csv,
                    file_name='rf.csv',
                    mime='text/csv',
                )
        elif model == "GBDT":
            if st.button('展示数据'):
                st.subheader("K折交叉验证结果:wavy_dash:")
                img = Image.open("C:/课程/o2o/figs/GBDT_cross_val.png")
                st.image(img, width=820)
                st.subheader("示例验证（dataset1训练，dataset2预测）:bulb:")
                img = Image.open("C:/课程/o2o/figs/GBDT.png")
                st.image(img, width=820)
                st.subheader("预测（对dataset3进行预测）:mag_right:")
                st.dataframe(gbdt.head(1000))
                st.download_button(
                    label="Download data as CSV",
                    data=gbdt_csv,
                    file_name='gbdt.csv',
                    mime='text/csv',
                )
        else:
            if st.button('展示数据'):
                st.subheader("K折交叉验证结果:wavy_dash:")
                img = Image.open("C:/课程/o2o/figs/XGB_cross_val.png")
                st.image(img, width=820)
                st.subheader("示例验证（dataset1训练，dataset2预测）:bulb:")
                img = Image.open("C:/课程/o2o/figs/XGB.png")
                st.image(img, width=820)
                st.subheader("预测（对dataset3进行预测）:mag_right:")
                st.dataframe(xgb.head(1000))
                st.download_button(
                    label="Download data as CSV",
                    data=xgb_csv,
                    file_name='xgb.csv',
                    mime='text/csv',
                )

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/


#     st.subheader("优惠券初始数据可视化")  # 副标题显示
#     type_of_data = st.selectbox("请选择想要展示的数据集",
#                                 ["train（训练集）", "test（测试集）"])
#     if st.button('展示数据'):
#         if type_of_data == "train（训练集）":
#             st.write(train.head(10))
#         elif type_of_data == "test（测试集）":
#             st.write(test.head(10))
#
#
#
#     def label(row):
#         if row['Date_received'] == 'null':
#             return -1
#         if row['Date'] != 'null':
#             td = pd.to_datetime(row['Date'], format='%Y%m%d') - pd.to_datetime(row['Date_received'], format='%Y%m%d')
#             if td <= pd.Timedelta(15, 'D'):
#                 return 1
#         return 0
#
#     train['label'] = train.apply(label, axis=1)  # 按列计算
#
#     train['sum'] = 1  # 给每一条记录付给一个初始值1,记录出现的次数
#     user_id_count = train.groupby(['User_id'], as_index=False)['sum'].agg({'count': np.sum})  # 统计各个用户出现的次数
#
#     def user_count(data):
#         if data > 10:
#             return 3
#         elif data > 5:
#             return 2
#         elif data > 1:
#             return 1
#         else:
#             return 0
#
#     st.subheader("用户活跃度调查")
#     user_id_count['user_range'] = user_id_count['count'].map(user_count)
#     f, ax = plt.subplots(1, 2, figsize=(17.5, 8))
#     user_id_count['user_range'].value_counts().plot(kind="pie", autopct='%1.1f%%', ax=ax[0], explode=[0.01, 0, 0, 0],
#                                                     startangle=90)
#     ax[0].set_title('user_range_ratio')
#     ax[0].set_ylabel('')
#     # user_id_count['user_range'].value_counts().plot(kind='bar',ax=ax[1])
#     sns.countplot('user_range', data=user_id_count, ax=ax[1])
#     ax[1].set_title('user range distribution')
#     st.pyplot(f)
#
#     train['sum'] = 1  # 给每一条记录付给一个初始值1,记录出现的次数
#     merchant_count = train.groupby(['Merchant_id'], as_index=False)['sum'].agg({'count': np.sum})  # 统计各个用户出现的次数
#
#     def Mer_count(data):
#         if data > 1000:
#             return 3
#         elif data > 100:
#             return 2
#         elif data > 20:
#             return 1
#         else:
#             return 0
#
#     st.subheader("商户活跃度调查")
#     merchant_count['mer_range'] = merchant_count['count'].map(Mer_count)  # 对商家也进行编码，3代表发放了很多的优惠券，2,1,0依次抵减
#     f, ax = plt.subplots(1, 2, figsize=(17.5, 8))
#     merchant_count['mer_range'].value_counts().plot(kind="pie", autopct='%1.1f%%', ax=ax[0], explode=[0.01, 0, 0, 0],
#                                                     startangle=90)
#     ax[0].set_title('mer_range')
#     ax[0].set_ylabel('')
#
#     sns.countplot('mer_range', data=merchant_count, ax=ax[1])
#     ax[1].set_title('merchant range distribution')
#     st.pyplot(f)
#
#     st.subheader("优惠券分布调查")
#     train1 = train[(train['Coupon_id'] != 'null')]
#     Cou_id_count = train1.groupby(['Coupon_id'], as_index=False)['sum'].agg({'count': np.sum})
#
#     def Cou_count(data):
#         if data > 1000:
#             return 3
#         elif data > 100:
#             return 2
#         elif data > 10:
#             return 1
#         else:
#             return 0
#
#     Cou_id_count['Cou_range'] = Cou_id_count['count'].map(Cou_count)
#     # 绘图
#     f, ax = plt.subplots(1, 2, figsize=(17.5, 8))
#     Cou_id_count['Cou_range'].value_counts().plot(kind="pie", autopct='%1.1f%%', ax=ax[0], explode=[0.01, 0, 0, 0],
#                                                   startangle=90)
#     ax[0].set_title('Cou_range')
#     ax[0].set_ylabel('')
#
#     sns.countplot('Cou_range', data=Cou_id_count, ax=ax[1])
#     ax[1].set_title('Cou_range distribution')
#     st.pyplot(f)
#
#     train1.Coupon_id = train1.Coupon_id.astype("int64")
#
#     st.subheader("优惠券优惠率分布")
#     def convertRate(row):
#         """Convert discount to rate"""
#         if row == 'null':
#             return 1.0
#         elif ':' in row:
#             rows = row.split(':')
#             return np.round(1.0 - float(rows[1]) / float(rows[0]), 2)
#         else:
#             return float(row)
#
#     train1['discount_rate'] = train1['Discount_rate'].apply(convertRate)
#     st.bar_chart((train1['discount_rate'].value_counts()/len(train)))
#
#     st.subheader("距离分布")
#     st.bar_chart((train['Distance'].value_counts()/len(train)))
#
#     st.subheader("优惠券使用情况分析")
#     couponbydate = train[train['Date_received'] != 'null'][['Date_received', 'Date']].groupby(['Date_received'],
#                                                                                               as_index=False).count()
#     couponbydate.columns = ['Date_received', 'count']
#     buybydate = train[(train['Date'] != 'null') & (train['Date_received'] != 'null')][
#         ['Date_received', 'Date']].groupby(['Date_received'], as_index=False).count()
#     buybydate.columns = ['Date_received', 'count']
#     date_buy = train['Date'].unique()  # 购物的日期
#     date_buy = sorted(date_buy[date_buy != 'null'])  # 按照购物的日期进行排，排除null值
#
#     date_received = train['Date_received'].unique()  # 接收到优惠券的时间
#     date_received = sorted(date_received[date_received != 'null'])  # 按照接收优惠券的时间排序
#
#     sns.set_style('ticks')
#     sns.set_context("notebook", font_scale=1.4)
#     f = plt.figure(figsize=(12, 8))
#     date_received_dt = pd.to_datetime(date_received, format='%Y%m%d')  # 转换为datetime格式
#     plt.subplot(211)
#     plt.bar(date_received_dt, couponbydate['count'], label='number of coupon received',
#             color='#a675a1')  # 绘制接收优惠券的日期对应的bar图
#     plt.bar(date_received_dt, buybydate['count'], label='number of coupon used', color='#75a1a6')  # 绘制使用优惠券对应的bar图
#     plt.yscale('log')
#     plt.ylabel('Count')
#     plt.legend()
#
#     plt.subplot(212)
#     plt.bar(date_received_dt, buybydate['count'] / couponbydate['count'], color='#62a5de')  # 绘制优惠券使用的比例图
#     plt.ylabel('Ratio(coupon used/coupon received)')
#     plt.tight_layout()
#     st.pyplot(f)
