import streamlit as st
import plotly.express as px

def render(df_raw, sg_col, val_col):
    st.header("🔍 1단계: 실시간 수집 공정 데이터 구조 분석")
    st.write("공정 데이터의 기초적인 형상과 그룹별 변동성을 시각적으로 확인합니다.")
    
    col1, col2 = st.columns([4, 6])
    with col1:
        st.subheader("데이터 데이터프레임 구조 요약")
        st.dataframe(df_raw, use_container_width=True, height=400)
        st.caption(f"전체 관측치 개수: {len(df_raw)}행 | 분석 대상 특성치: {val_col}")
    
    with col2:
        st.subheader("부품/라인별 단기 산포 분포 박스플롯")
        fig_box = px.box(
            df_raw, x=sg_col, y=val_col, color=sg_col, points='all',
            title=f'[{val_col}] 그룹별 변동성 조감', template='seaborn'
        )
        fig_box.update_layout(margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_box, use_container_width=True)
