import streamlit as st
import pandas as pd
import stats_engine as engine
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def render(df_raw, sg_col, val_col):
    st.header("📉 계량형 Shewhart 변수 제어 관리도")
    st.write("우연 원인과 이상 원인을 식별 차단하여 공정을 통계적 안정 상태로 유도합니다.")
    
    chart_col1, chart_col2 = st.columns([4, 6])
    with chart_col1:
        chart_mode = st.selectbox("관리도 유형 선택", ["Xbar-R", "Xbar-s", "I-MR"])
        window_param = st.slider("이동 범위 분석 윈도우 크기(w)", 2, 10, 3) if chart_mode == "I-MR" else 3
        
    chart1, chart2 = engine.generate_value_chart_data(df_raw, sg_col, val_col, chart_type=chart_mode, window=window_param)
    ooc_points = chart1[(chart1['point'] > chart1['UCL']) | (chart1['point'] < chart1['LCL'])].index.tolist()
    
    with chart_col2:
        if ooc_points:
            st.error(f"🚨 [이상 원인 포착] 관리 한계를 이탈한 이상 변동 부분군 발견: {ooc_points}")
            exclude_ooc = st.checkbox("이상 부분군 제거 후 관리 한계선 재산출 반영")
            if exclude_ooc:
                cleaned_df = df_raw[~df_raw[sg_col].isin(ooc_points)].copy()
                chart1, chart2 = engine.generate_value_chart_data(cleaned_df, sg_col, val_col, chart_type=chart_mode, window=window_param)
                st.success("🔄 정제 완료: 부적합 변동 원인을 제외한 새로운 안정 관리선이 적용되었습니다.")
        else:
            st.success("🎯 모든 공정 계측점이 관리 한계선 내부에서 안정적으로 구동 중입니다.")

    names = chart_mode.split('-')
    sub1, sub2 = names[0], names[1] if len(names) > 1 else 'MR'
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=(f'{sub1} 관리도', f'{sub2} 관리도'))
    
    # [수정 적용]: row=1, row=2 레이어의 엠베딩 점선 다중 믹싱 버그 차단 고정 배치
    for idx, c_df in enumerate([chart1, chart2], start=1):
        color = 'royalblue' if idx == 1 else 'purple'
        fig.add_trace(go.Scatter(x=c_df.index, y=c_df['point'], mode='lines+markers', name=sub1 if idx==1 else sub2, marker=dict(color=color)), row=idx, col=1)
        fig.add_trace(go.Scatter(x=c_df.index, y=c_df['CL'], mode='lines', line=dict(color='green', dash='dashdot'), showlegend=False), row=idx, col=1)
        fig.add_trace(go.Scatter(x=c_df.index, y=c_df['LCL'], mode='lines', line=dict(color='red', dash='dot'), showlegend=False), row=idx, col=1)
        fig.add_trace(go.Scatter(x=c_df.index, y=c_df['UCL'], mode='lines', line=dict(color='magenta', dash='dot'), showlegend=False), row=idx, col=1)
        
        last = c_df.index[-1]
        fig.add_annotation(x=last, y=c_df['UCL'].iloc[-1], text=f"UCL={c_df['UCL'].iloc[-1]:.2f}", showarrow=False, font=dict(color='magenta'), row=idx, col=1, xshift=35)
        fig.add_annotation(x=last, y=c_df['CL'].iloc[-1], text=f"CL={c_df['CL'].iloc[-1]:.2f}", showarrow=False, font=dict(color='green'), row=idx, col=1, xshift=35)
        fig.add_annotation(x=last, y=c_df['LCL'].iloc[-1], text=f"LCL={c_df['LCL'].iloc[-1]:.2f}", showarrow=False, font=dict(color='red'), row=idx, col=1, xshift=35)

    fig.update_layout(height=550, showlegend=False, template='seaborn')
    st.plotly_chart(fig, use_container_width=True)
