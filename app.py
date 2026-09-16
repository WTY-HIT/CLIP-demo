import os
# 环境配置必须放在所有 import 之前
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
from transformers import pipeline
from PIL import Image

# 环境配置：依然是那两道保命符
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# --- 网页界面设计 ---
st.set_page_config(page_title="CLIP 零样本分类器", layout="centered")
st.title("🤖 CLIP 零样本图像分类器")
st.write("上传一张图片，输入你想区分的候选标签（用英文逗号隔开），看看 CLIP 怎么选！")


# 使用缓存，保证模型只加载一次，不会因为网页刷新而重新加载
@st.cache_resource
def load_model():
    return pipeline("zero-shot-image-classification", model="openai/clip-vit-base-patch32")


# 加载模型
with st.spinner("正在加载模型，请稍候..."):
    classifier = load_model()

# 1. 上传图片组件
uploaded_file = st.file_uploader("请上传一张图片 (JPG/PNG)", type=["jpg", "jpeg", "png"])

# 2. 输入标签组件（默认填入你常用的标签）
default_labels = "a road with cars, a sidewalk, a building, a tree"
labels_input = st.text_input("请输入候选标签（用英文逗号隔开）:", value=default_labels)

# 3. 开始检测按钮
if st.button("开始识别", type="primary"):
    if uploaded_file is not None:
        # 把上传的图片转成 PIL 格式
        image = Image.open(uploaded_file)
        # 展示图片
        st.image(image, caption="你上传的图片", use_container_width=True)

        # 处理标签：去掉空格，按逗号分割
        labels = [label.strip() for label in labels_input.split(",")]

        # 进行识别
        with st.spinner("CLIP 正在思考..."):
            results = classifier(image, candidate_labels=labels)

        # 在网页上展示结果
        st.success("识别完成！")
        for result in results:
            # 进度条展示置信度
            score = result['score']
            st.write(f"**标签：{result['label']}**")
            st.progress(score)
            st.write(f"置信度：{score:.4f}")

    else:
        st.warning("请先上传一张图片哦！")