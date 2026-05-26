import streamlit as st
import plotly.express as px

def render(df_raw, sg_col, val_col):
    st.header("🔍 공정 데이터 형상 및 기초 시각화")
    st.write("공정 전체 데이터의 흐름 패턴과 부분군 간의 산포 분포를 종합 조감합니다.")
    
    col1, col2 = st.columns([4, 6])
    with col1:
        st.subheader("📋 데이터프레임 구조 요약")
        st.dataframe(df_raw, use_container_width=True, height=350)
        st.caption(f"전체 관측치 개수: {len(df_raw)}행 | 분석 대상 변수: {val_col}")
    
    with col2:
        st.subheader("📦 그룹별 단기 산포 분포 (Boxplot)")
        fig_box = px.box(df_raw, x=sg_col, y=val_col, color=sg_col, points='all', template='seaborn')
        fig_box.update_layout(height=350, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_box, use_container_width=True)
        
    st.subheader("📈 실시간 품질 계측치 시계열 추세 분석 (Time-Series Plot)")
    fig_line = px.scatter(df_raw, x=df_raw.index, y=val_col, color=sg_col, title=f"시간 경과에 따른 {val_col}의 공정 거동", template='seaborn')
    fig_line.update_traces(mode='lines+markers')
    fig_line.update_layout(height=350)
    st.plotly_chart(fig_line, use_container_width=True)
