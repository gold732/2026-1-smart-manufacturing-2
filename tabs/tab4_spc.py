import streamlit as st
import pandas as pd
import stats_engine as engine
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def render(df_raw, sg_col, val_col):
    st.header("📉 4단계: Shewhart 제어 모니터링 관리도 (SPC)")
    st.write("우연 원인과 이상 원인을 선별 차단하여 공정을 관리상태로 제어합니다.")
    
    chart_col1, chart_col2 = st.columns([4, 6])
    with chart_col1:
        chart_mode = st.selectbox("품질 관리 기법도 유형 선택", ["Xbar-R", "Xbar-s", "I-MR"])
        window_param = st.slider("Individual Moving Range 이동 분석 윈도우 크기(w)", 2, 10, 3) if chart_mode == "I-MR" else 3
        
    chart1, chart2 = engine.generate_value_chart_data(df_raw, sg_col, val_col, chart_type=chart_mode, window=window_param)
    ooc_points = chart1[(chart1['point'] > chart1['UCL']) | (chart1['point'] < chart1['LCL'])].index.tolist()
    
    with chart_col2:
        if ooc_points:
            st.error(f"🚨 [이상 변동 시점 포착]: 관리 한계 한계치를 탈출한 노드가 존재합니다: {ooc_points}")
            exclude_ooc = st.checkbox("이상 부분군(Lot) 원인 데이터 정제 제거 후 제어선 재산출 가동")
            if exclude_ooc:
                cleaned_df = df_raw[~df_raw[sg_col].isin(ooc_points)].copy()
                chart1, chart2 = engine.generate_value_chart_data(cleaned_df, sg_col, val_col, chart_type=chart_mode, window=window_param)
                st.success(f"정제 가동 완료: 총 {len(df_raw)-len(cleaned_df)}건의 노드를 격리 제거하고 한계선을 타이트하게 재구성했습니다.")
        else:
            st.success("🎯 공정이 오직 우연 원인에 의해서만 움직이는 대단히 안정된 상태입니다.")

    # 인터랙티브 모니터링 대시보드 드로잉
    names = chart_mode.split('-')
    sub1, sub2 = names[0], names[1] if len(names) > 1 else 'MR'
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=(f'{sub1} 관리도', f'{sub2} 관리도'))
    
    # 상단 챠트 데이터 바인딩
    fig.add_trace(go.Scatter(x=chart1.index, y=chart1['point'], mode='lines+markers', name=sub1), row=1, col=1)
    fig.add_trace(go.Scatter(x=chart1.index, y=chart1['CL'], mode='lines', line=dict(color='green', dash='dashdot'), name='CL'), row=1, col=1)
    fig.add_trace(go.Scatter(x=chart1.index, y=chart1['LCL'], mode='lines', line=dict(color='red', dash='dot'), name='LCL'), row=1, col=1)
    fig.add_trace(go.Scatter(x=chart1.index, y=chart1['UCL'], mode='lines', line=dict(color='magenta', dash='dot'), name='UCL'), row=1, col=1)
    
    # 하단 챠트 데이터 바인딩
    fig.add_trace(go.Scatter(x=chart2.index, y=chart2['point'], mode='lines+markers', name=sub2, marker=dict(color='purple')), row=2, col=1)
    fig.add_trace(go.Scatter(x=chart2.index, y=chart2['CL'], mode='lines', line=dict(color='green', dash='dashdot'), name='CL'), row=2, col=1)
    fig.add_trace(go.Scatter(x=chart2.index, y=chart2['LCL'], mode='lines', line=dict(color='red', dash='dot'), name='LCL'), row=2, col=1)
    fig.add_trace(go.Scatter(x=chart2.index, y=chart2['UCL'], mode='lines', line=dict(color='magenta', dash='dot'), name='UCL'), row=2, col=1)
    
    # 주석 라벨 표시부
    for i, c_data in enumerate([chart1, chart2], start=1):
        last_idx = c_data.index[-1]
        fig.add_annotation(x=last_idx, y=c_data['UCL'].iloc[-1], text=f"UCL={c_data['UCL'].iloc[-1]:.2f}", showarrow=False, font=dict(color='magenta'), row=i, col=1, xshift=35)
        fig.add_annotation(x=last_idx, y=c_data['CL'].iloc[-1], text=f"CL={c_data['CL'].iloc[-1]:.2f}", showarrow=False, font=dict(color='green'), row=i, col=1, xshift=35)
        fig.add_annotation(x=last_idx, y=c_data['LCL'].iloc[-1], text=f"LCL={c_data['LCL'].iloc[-1]:.2f}", showarrow=False, font=dict(color='red'), row=i, col=1, xshift=35)

    fig.update_layout(height=650, title_text=f"{chart_mode} 공정 모니터링 시스템", showlegend=False, template='seaborn')
    st.plotly_chart(fig, use_container_width=True)
