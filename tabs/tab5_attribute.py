import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import stats_engine as engine

def render(df_raw, sg_col, val_col):
    st.header("🏷️ 계수형 불량 및 결함 제어 차트")
    st.write("부적합품 개수 및 결함 건수 속성을 통계적으로 선별하여 모니터링합니다.")
    
    attr_mode = st.selectbox("계수형 관리도 유형 선택", ["P 차트 (불량률)", "NP 차트 (불량개수)", "C 차트 (결점수)", "U 차트 (단위당 결점수)"])
    
    # 💡 [보완 핵심] 수식 엔진 내부 집계 결과를 직접 받아와 동적 가변 관리선 바인딩 처리
    chart_type_map = {"P 차트 (불량률)": "P", "NP 차트 (불량개수)": "NP", "C 차트 (결점수)": "C", "U 차트 (단위당 결점수)": "U"}
    target_mode = chart_type_map[attr_mode]
    
    res_df = engine.generate_attribute_chart_data(df_raw, sg_col, val_col, chart_type=target_mode)
    
    fig = go.Figure()
    
    # 공정 데이터 시계열 포인트 매핑
    color_map = {"P": "darkorange", "NP": "crimson", "C": "indigo", "U": "teal"}
    fig.add_trace(go.Scatter(x=res_df.index, y=res_df['point'], mode='lines+markers', name='측정치', marker=dict(color=color_map[target_mode])))
    
    # 가변 표본 크기에 따라 톱니처럼 동적으로 꺾이는 상하한 관리선 정밀 렌더링
    fig.add_trace(go.Scatter(x=res_df.index, y=res_df['UCL'], mode='lines', line=dict(color='magenta', dash='dot'), name='UCL'))
    fig.add_trace(go.Scatter(x=res_df.index, y=res_df['LCL'], mode='lines', line=dict(color='red', dash='dot'), name='LCL'))
    
    # 중심선 고정 레이아웃
    cl_mean = res_df['CL'].mean()
    fig.add_hline(y=cl_mean, line_color='green', line_dash='dashdot', annotation_text=f"CL ({cl_mean:.4f})")
    
    fig.update_layout(title=f"계수형 품질 모니터링: {attr_mode} 분석 결과 리포트", template='seaborn', height=420)
    st.plotly_chart(fig, use_container_width=True)
