import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

def render(df_raw, sg_col, val_col, lsl, usl, sigma_method):
    st.header("🏷️ 5단계 대시보드: 계수형 불량/결함 전용 SPC 제어 차트 스크린")
    st.write("강의록 2페이지 및 9~11페이지에 근거하여 불량 속성을 모니터링합니다.")
    
    attr_mode = st.selectbox("계수형 불량 판단 관리도 아키텍처 지정", ["P 차트 (불량률 추적)", "NP 차트 (불량개수 추적)", "C 차트 (결점수 추적)", "U 차트 (단위당 결점수 추적)"])
    
    np.random.seed(42)
    size_sim = len(df_raw)
    sample_sizes = np.random.randint(200, 350, size=size_sim)
    
    if "P" in attr_mode or "NP" in attr_mode:
        defects = np.random.binomial(n=sample_sizes, p=0.02)
        p_bar = defects.sum() / sample_sizes.sum()
        fig = go.Figure()
        if "NP" in attr_mode:
            np_bar = defects.mean()
            ucl = np_bar + 3 * np.sqrt(np_bar * (1 - p_bar))
            lcl = max(0, np_bar - 3 * np.sqrt(np_bar * (1 - p_bar)))
            fig.add_trace(go.Scatter(x=df_raw[sg_col], y=defects, mode='lines+markers', marker=dict(color='crimson')))
            cl_val, ucl_val, lcl_val = np_bar, ucl, lcl
        else:
            p_i = defects / sample_sizes
            ucl = p_bar + 3 * np.sqrt(p_bar * (1 - p_bar) / sample_sizes)
            lcl = np.clip(p_bar - 3 * np.sqrt(p_bar * (1 - p_bar) / sample_sizes), 0, 1)
            fig.add_trace(go.Scatter(x=df_raw[sg_col], y=p_i, mode='lines+markers', marker=dict(color='darkorange')))
            cl_val, ucl_val, lcl_val = p_bar, ucl, lcl
    else:
        defects = np.random.poisson(lam=5, size=size_sim)
        c_bar = defects.mean()
        fig = go.Figure()
        if "C" in attr_mode:
            ucl = c_bar + 3 * np.sqrt(c_bar)
            lcl = max(0, c_bar - 3 * np.sqrt(c_bar))
            fig.add_trace(go.Scatter(x=df_raw[sg_col], y=defects, mode='lines+markers', marker=dict(color='indigo')))
            cl_val, ucl_val, lcl_val = c_bar, ucl, lcl
        else:
            u_i = defects / 5
            u_bar = defects.sum() / (size_sim * 5)
            ucl = u_bar + 3 * np.sqrt(u_bar / 5)
            lcl = max(0, u_bar - 3 * np.sqrt(u_bar / 5))
            fig.add_trace(go.Scatter(x=df_raw[sg_col], y=u_i, mode='lines+markers', marker=dict(color='teal')))
            cl_val, ucl_val, lcl_val = u_bar, ucl, lcl

    fig.add_hline(y=np.mean(cl_val), line_color='green', line_dash='dashdot', annotation_text=f"CL ({np.mean(cl_val):.4f})")
    fig.add_hline(y=np.mean(ucl_val), line_color='magenta', line_dash='dot', annotation_text=f"UCL ({np.mean(ucl_val):.4f})")
    fig.add_hline(y=np.mean(lcl_val), line_color='red', line_dash='dot', annotation_text=f"LCL ({np.mean(lcl_val):.4f})")
    
    fig.update_layout(title=f"계수형 품질 모니터링: {attr_mode} 가동 결과", template='seaborn', height=450)
    st.plotly_chart(fig, use_container_width=True)
