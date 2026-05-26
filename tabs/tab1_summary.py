import streamlit as st
import plotly.express as px

def render(df_raw, sg_col, val_col):
    st.header("🔍 1단계 대시보드: 수집 데이터 형상 및 기초 시각화")
    st.write("공정 전체 데이터의 흐름 패턴과 부분군 간의 산포 분포를 종합 조감합니다.")
    
    col1, col2 = st.columns([4, 6])
    with col1:
        st.subheader("📋 수집 데이터프레임 요약")
        st.dataframe(df_raw, use_container_width=True, height=400)
        st.caption(f"가용 총 데이터 개수: {len(df_raw)}행 | 모니터링 품질인자: {val_col}")
    
    with col2:
        st.subheader("📦 그룹별 단기 산포 분포 (Boxplot)")
        fig_box = px.box(df_raw, x=sg_col, y=val_col, color=sg_col, points='all', template='seaborn')
        st.plotly_chart(fig_box, use_container_width=True)
        
    st.subheader("📈 실시간 품질 계측치 시계열 추세 분석 (Time-Series Plot)")
    fig_line = px.scatter(df_raw, x=df_raw.index, y=val_col, color=sg_col, markers=True, title=f"시간 경과에 따른 {val_col}의 공정 고유 거동", template='seaborn')
    fig_line.update_traces(mode='lines+markers')
    st.plotly_chart(fig_line, use_container_width=True)
    
