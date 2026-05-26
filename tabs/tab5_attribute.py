import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

def render(df_raw, sg_col, val_col):
    st.header("🏷️ 계수형 불량 및 결함 제어 차트")
    st.write("부적합품 개수 및 결함 건수 속성을 통계적으로 선별하여 모니터링합니다.")
    
    attr_mode = st.selectbox("계수형 관리도 유형 선택", ["P 차트 (불량률)", "NP 차트 (불량개수)", "C 차트 (결점수)", "U 차트 (단위당 결점수)"])
    
    np.random.seed(42)
    size_sim = len(df_raw)
    # 가변 표본 크기 패턴 시뮬레이션 적용
    sample_sizes = np.random.randint(200, 350, size=size_sim)
    defects = np.random.binomial(n=sample_sizes, p=0.02)
    p_bar = defects.sum() / sample_sizes.sum()
    
    fig = go.Figure()
    
    if "P 차트" in attr_mode:
        p_i = defects / sample_sizes
        # 가변 n_i 반영한 계수형 통계 한계선 수식 매핑 완료
        ucl = p_bar + 3 * np.sqrt(p_bar * (1 - p_bar) / sample_sizes)
        lcl = np.clip(p_bar - 3 * np.sqrt(p_bar * (1 - p_bar) / sample_sizes), 0, 1)
        fig.add_trace(go.Scatter(x=df_raw[sg_col], y=p_i, mode='lines+markers', name='불량률', marker=dict(color='darkorange')))
        fig.add_trace(go.Scatter(x=df_raw[sg_col], y=ucl, mode='lines', line=dict(color='magenta', dash='dot'), name='UCL'))
        fig.add_trace(go.Scatter(x=df_raw[sg_col], y=lcl, mode='lines', line=dict(color='red', dash='dot'), name='LCL'))
        fig.add_hline(y=p_bar, line_color='green', line_dash='dashdot', annotation_text=f"CL ({p_bar:.4f})")
        
    elif "NP 차트" in attr_mode:
        np_bar = defects.mean()
        ucl = np_bar + 3 * np.sqrt(np_bar * (1 - p_bar))
        lcl = max(0, np_bar - 3 * np.sqrt(np_bar * (1 - p_bar)))
        fig.add_trace(go.Scatter(x=df_raw[sg_col], y=defects, mode='lines+markers', name='불량개수', marker=dict(color='crimson')))
        fig.add_hline(y=ucl, line_color='magenta', line_dash='dot', annotation_text=f"UCL ({ucl:.2f})")
        fig.add_hline(y=lcl, line_color='red', line_dash='dot', annotation_text=f"LCL ({lcl:.2f})")
        fig.add_hline(y=np_bar, line_color='green', line_dash='dashdot', annotation_text=f"CL ({np_bar:.2f})")
        
    elif "C 차트" in attr_mode:
        c_vals = np.random.poisson(lam=5, size=size_sim)
        c_bar = c_vals.mean()
        ucl = c_bar + 3 * np.sqrt(c_bar)
        lcl = max(0, c_bar - 3 * np.sqrt(c_bar))
        fig.add_trace(go.Scatter(x=df_raw[sg_col], y=c_vals, mode='lines+markers', name='결점수', marker=dict(color='indigo')))
        fig.add_hline(y=ucl, line_color='magenta', line_dash='dot', annotation_text=f"UCL ({ucl:.2f})")
        fig.add_hline(y=lcl, line_color='red', line_dash='dot', annotation_text=f"LCL ({lcl:.2f})")
        fig.add_hline(y=c_bar, line_color='green', line_dash='dashdot', annotation_text=f"CL ({c_bar:.2f})")
        
    elif "U 차트" in attr_mode:
        c_vals = np.random.poisson(lam=5, size=size_sim)
        u_i = c_vals / 5
        u_bar = c_vals.sum() / (size_sim * 5)
        ucl = u_bar + 3 * np.sqrt(u_bar / 5)
        lcl = max(0, u_bar - 3 * np.sqrt(u_bar / 5))
        fig.add_trace(go.Scatter(x=df_raw[sg_col], y=u_i, mode='lines+markers', name='단위당 결점수', marker=dict(color='teal')))
        fig.add_hline(y=ucl, line_color='magenta', line_dash='dot', annotation_text=f"UCL ({ucl:.4f})")
        fig.add_hline(y=lcl, line_color='red', line_dash='dot', annotation_text=f"LCL ({lcl:.4f})")
        fig.add_hline(y=u_bar, line_color='green', line_dash='dashdot', annotation_text=f"CL ({u_bar:.4f})")

    fig.update_layout(title=f"계수형 품질 변동 추정 가동 결과", template='seaborn', height=400)
    st.plotly_chart(fig, use_container_width=True)
