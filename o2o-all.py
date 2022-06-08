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
    #img = Image.open("C:/课程/o2o/figs/1.png")
    st.image("https://github.com/ZanyPink/biye/blob/main/photo/1.png", width=820)

    @st.cache(allow_output_mutation=True)
    def load_pre_train():
        train_data = pd.read_csv("https://github.com/ZanyPink/biye/blob/main/data/ccf_offline_stage1_train.csv", encoding='utf-8', sep='\t')
        return train_data

    @st.cache(allow_output_mutation=True)
    def load_pre_test():
        test_data = pd.read_csv("https://github.com/ZanyPink/biye/blob/main/data/ccf_offline_stage1_test_revised.csv", encoding='utf-8', sep='\t')
        return test_data

    @st.cache(allow_output_mutation=True)
    def load_pre_online():
        online_data = pd.read_csv("https://github.com/ZanyPink/biye/blob/main/data/ccf_online_stage1_train.csv", encoding='utf-8', sep='\t')
        return online_data

    @st.cache(allow_output_mutation=True)
    def load_rf():
        rf_data = pd.read_csv("https://github.com/ZanyPink/biye/blob/main/result/rf_preds.csv", encoding='utf-8', sep='\t')
        return rf_data

    @st.cache(allow_output_mutation=True)
    def load_gbdt():
        gbdt_data = pd.read_csv("https://github.com/ZanyPink/biye/blob/main/result/gbdt_preds.csv", encoding='utf-8', sep='\t')
        return gbdt_data

    @st.cache(allow_output_mutation=True)
    def load_xgb():
        xgb_data = pd.read_csv("https://github.com/ZanyPink/biye/blob/main/result/xgb_preds.csv", encoding='utf-8', sep='\t')
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
                    #img = Image.open("C:/课程/o2o/figs/user_count.png")
                    st.image("https://github.com/ZanyPink/biye/blob/main/photo/user_count.png", width=820)
                    st.caption("0（流失用户）：仅领取过一次优惠券")
                    st.caption("1（低活跃度用户）：领取优惠券2-5次")
                    st.caption("2（中活跃度用户）：领取优惠券6-10次")
                    st.caption("3（高活跃度用户）：领取优惠券超过10次")
            elif data_analyze == "商户活跃度":
                if st.button('展示数据'):
                    st.subheader("商户活跃度展示:bar_chart:")
                    #img = Image.open("C:/课程/o2o/figs/mer_count.png")
                    st.image("https://github.com/ZanyPink/biye/blob/main/photo/mer_count.png", width=820)
                    st.caption("0：商家被领取优惠券次数低于20")
                    st.caption("1：商家被领取优惠券20-100次")
                    st.caption("2：商家被领取优惠券100-1000次")
                    st.caption("3：商家被领取优惠券超过1000次")
            elif data_analyze == "优惠券概况":
                if st.button('展示数据'):
                    st.subheader("优惠券活跃度展示:bar_chart:")
                    #img = Image.open("C:/课程/o2o/figs/cou_count.png")
                    st.image("https://github.com/ZanyPink/biye/blob/main/photo/cou_count.png", width=820)
                    st.caption("0：优惠券被领取次数低于10")
                    st.caption("1：优惠券被领取11-100次")
                    st.caption("2：优惠券被领取101-1000")
                    st.caption("3：优惠券被领取超过1000次")
                    st.subheader("优惠券折扣率分布:chart_with_upwards_trend:")
                    #img = Image.open("C:/课程/o2o/figs/discount_rate.png")
                    st.image("https://github.com/ZanyPink/biye/blob/main/photo/discount_rate.png", width=820)
                    st.subheader("优惠券使用分布:chart_with_downwards_trend:")
                    #img = Image.open("C:/课程/o2o/figs/coupon_use.png")
                    st.image("https://github.com/ZanyPink/biye/blob/main/photo/coupon_use.png", width=820)
                    st.subheader("优惠券领取周分布:bar_chart:")
                    #img = Image.open("C:/课程/o2o/figs/cou_week.png")
                    st.image("https://github.com/ZanyPink/biye/blob/main/photo/cou_week.png", width=820)
                    st.subheader("优惠券领取月分布:bar_chart:")
                    #img = Image.open("C:/课程/o2o/figs/cou_month.png")
                    st.image("https://github.com/ZanyPink/biye/blob/main/photo/cou_month.png", width=820)
            else:
                if st.button('展示数据'):
                    st.subheader("距离分布:bar_chart:")
                    #img = Image.open("C:/课程/o2o/figs/distance from merchant.png")
                    st.image("https://github.com/ZanyPink/biye/blob/main/photo/distance%20from%20merchant.png", width=820)
        elif data_select == 'ccf_online_stage1_train.csv':
            if data_analyze == "用户活跃度":
                if st.button('展示数据'):
                    st.subheader("用户活跃度展示:bar_chart:")
                    #img = Image.open("C:/课程/o2o/figs/on_user_count.png")
                    st.image("https://github.com/ZanyPink/biye/blob/main/photo/on_user_count.png", width=820)
                    st.caption("0（流失用户）：仅领取过一次优惠券")
                    st.caption("1（低活跃度用户）：领取优惠券2-5次")
                    st.caption("2（中活跃度用户）：领取优惠券6-10次")
                    st.caption("3（高活跃度用户）：领取优惠券超过10次")
            elif data_analyze == "商户活跃度":
                if st.button('展示数据'):
                    st.subheader("商户活跃度展示:bar_chart:")
                    #img = Image.open("C:/课程/o2o/figs/on_mer_count.png")
                    st.image("https://github.com/ZanyPink/biye/blob/main/photo/on_mer_count.png", width=820)
                    st.caption("0：商家被领取优惠券次数低于20")
                    st.caption("1：商家被领取优惠券20-100次")
                    st.caption("2：商家被领取优惠券100-1000次")
                    st.caption("3：商家被领取优惠券超过1000次")
            elif data_analyze == "优惠券概况":
                if st.button('展示数据'):
                    st.subheader("优惠券活跃度展示:bar_chart:")
                    #img = Image.open("C:/课程/o2o/figs/on_cou_count.png")
                    st.image("https://github.com/ZanyPink/biye/blob/main/photo/on_cou_count.png", width=820)
                    st.caption("0：优惠券被领取次数低于10")
                    st.caption("1：优惠券被领取11-100次")
                    st.caption("2：优惠券被领取101-1000")
                    st.caption("3：优惠券被领取超过1000次")
                    st.subheader("优惠券折扣率分布:chart_with_upwards_trend:")
                    #img = Image.open("C:/课程/o2o/figs/on_discount_rate.png")
                    st.image("https://github.com/ZanyPink/biye/blob/main/photo/on_discount_rate.png", width=820)
                    st.subheader("优惠券使用分布:chart_with_downwards_trend:")
                    #img = Image.open("C:/课程/o2o/figs/on_coupon_use.png")
                    st.image("https://github.com/ZanyPink/biye/blob/main/photo/on_coupon_use.png", width=820)
            else:
                if st.button('展示数据'):
                    st.error("线上数据集没有距离字段:worried:")
                    st.snow()
        else:
            if data_analyze == "用户活跃度":
                if st.button('展示数据'):
                    st.subheader("用户活跃度展示:bar_chart:")
                    #img = Image.open("C:/课程/o2o/figs/test_user_count.png")
                    st.image("https://github.com/ZanyPink/biye/blob/main/photo/test_user_count.png", width=820)
                    st.caption("0（流失用户）：仅领取过一次优惠券")
                    st.caption("1（低活跃度用户）：领取优惠券2-5次")
                    st.caption("2（中活跃度用户）：领取优惠券6-10次")
                    st.caption("3（高活跃度用户）：领取优惠券超过10次")
            elif data_analyze == "商户活跃度":
                if st.button('展示数据'):
                    st.subheader("商户活跃度展示:bar_chart:")
                    #img = Image.open("C:/课程/o2o/figs/test_mer_count.png")
                    st.image("https://github.com/ZanyPink/biye/blob/main/photo/test_mer_count.png", width=820)
                    st.caption("0：商家被领取优惠券次数低于20")
                    st.caption("1：商家被领取优惠券20-100次")
                    st.caption("2：商家被领取优惠券100-1000次")
                    st.caption("3：商家被领取优惠券超过1000次")
            elif data_analyze == "优惠券概况":
                if st.button('展示数据'):
                    st.subheader("优惠券活跃度展示:bar_chart:")
                    #img = Image.open("C:/课程/o2o/figs/test_cou_count.png")
                    st.image("https://github.com/ZanyPink/biye/blob/main/photo/test_cou_count.png", width=820)
                    st.caption("0：优惠券被领取次数低于10")
                    st.caption("1：优惠券被领取11-100次")
                    st.caption("2：优惠券被领取101-1000")
                    st.caption("3：优惠券被领取超过1000次")
                    st.subheader("优惠券折扣率分布:chart_with_upwards_trend:")
                    #img = Image.open("C:/课程/o2o/figs/test_discount_rate.png")
                    st.image("https://github.com/ZanyPink/biye/blob/main/photo/test_discount_rate.png", width=820)
            else:
                if st.button('展示数据'):
                    st.subheader("距离分布:bar_chart:")
                    #img = Image.open("C:/课程/o2o/figs/test distance from merchant.png")
                    st.image("https://github.com/ZanyPink/biye/blob/main/photo/test%20distance%20from%20merchant.png", width=820)


    elif select == "特征热图":
        heatmap = st.selectbox("请选择热图", ["全部特征热图", "上采样热图", "PCA及热图"])
        if heatmap == "全部特征热图":
            if st.button('展示数据'):
                st.subheader("全部特征热图:art:")
                #img = Image.open("C:/课程/o2o/figs/heatmap_all.png")
                st.image("https://github.com/ZanyPink/biye/blob/main/photo/heatmap_all.png", width=820)
        elif heatmap == "上采样热图":
            if st.button('展示数据'):
                st.subheader("上采样热图:art:")
                #img = Image.open("C:/课程/o2o/figs/heatmap_after.png")
                st.image("https://github.com/ZanyPink/biye/blob/main/photo/heatmap_after.png", width=820)
        else:
            if st.button('展示数据'):
                st.subheader("PCA:bar_chart:")
                #img = Image.open("C:/课程/o2o/figs/PCA.png")
                st.image("https://github.com/ZanyPink/biye/blob/main/photo/PCA.png", width=820)
                st.subheader("热图:art:")
                #img = Image.open("C:/课程/o2o/figs/heatmap_after.png")
                st.image("https://github.com/ZanyPink/biye/blob/main/photo/heatmap_after.png", width=820)
    elif select == "预测结果":
        model = st.selectbox("请选择使用的模型", ["Random Forest", "GBDT", "XGBoost"])
        if model == "Random Forest":
            if st.button('展示数据'):
                st.subheader("K折交叉验证结果:wavy_dash:")
                #img = Image.open("C:/课程/o2o/figs/RF_cross_val.png")
                st.image("https://github.com/ZanyPink/biye/blob/main/photo/RF_cross_val.png", width=820)
                st.subheader("示例验证（dataset1训练，dataset2预测）:bulb:")
                #img = Image.open("C:/课程/o2o/figs/RF.png")
                st.image("https://github.com/ZanyPink/biye/blob/main/photo/RF.png", width=820)
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
                #img = Image.open("C:/课程/o2o/figs/GBDT_cross_val.png")
                st.image("https://github.com/ZanyPink/biye/blob/main/photo/GBDT_cross_val.png", width=820)
                st.subheader("示例验证（dataset1训练，dataset2预测）:bulb:")
                #img = Image.open("C:/课程/o2o/figs/GBDT.png")
                st.image("https://github.com/ZanyPink/biye/blob/main/photo/GBDT.png", width=820)
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
                #img = Image.open("C:/课程/o2o/figs/XGB_cross_val.png")
                st.image("https://github.com/ZanyPink/biye/blob/main/photo/XGB_cross_val.png", width=820)
                st.subheader("示例验证（dataset1训练，dataset2预测）:bulb:")
                #img = Image.open("C:/课程/o2o/figs/XGB.png")
                st.image("https://github.com/ZanyPink/biye/blob/main/photo/XGB.png", width=820)
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
