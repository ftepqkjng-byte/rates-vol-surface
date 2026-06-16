import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(layout="wide", page_title="Volatility Dashboard")

EXPIRIES = ['1Y', '2Y', '3Y', '5Y', '7Y', '10Y', '15Y', '20Y']
TENORS = ['5Y', '10Y', '20Y']

MASK = pd.DataFrame(False, index=EXPIRIES, columns=TENORS)
MASK.loc[['3Y', '7Y', '10Y', '15Y', '20Y'], '5Y'] = True
MASK.loc[['15Y', '20Y'], '20Y'] = True


def apply_mask(df: pd.DataFrame) -> pd.DataFrame:
    df[MASK] = np.nan
    return df


def make_spot_vol(seed: int) -> pd.DataFrame:
    data = np.random.default_rng(seed).uniform(4.5, 5.8, (8, 3))
    return apply_mask(pd.DataFrame(data, index=EXPIRIES, columns=TENORS))


def make_fwd_vol(seed: int) -> pd.DataFrame:
    data = np.random.default_rng(seed).uniform(4.6, 5.9, (8, 3))
    return apply_mask(pd.DataFrame(data, index=EXPIRIES, columns=TENORS))


def make_ratio(spot: pd.DataFrame, fwd: pd.DataFrame) -> pd.DataFrame:
    return (fwd / spot).round(2)


def make_ranking(ratio: pd.DataFrame) -> pd.DataFrame:
    valid = ratio.stack()
    ranks = valid.rank(method='min', ascending=True).astype(int)
    result = ratio.copy() * np.nan
    for (row, col), r in ranks.items():
        result.loc[row, col] = r
    return result


def color_ranking(df: pd.DataFrame):
    valid = df.stack()
    if valid.empty:
        return df.map(lambda _: '')
    min_r, max_r = valid.min(), valid.max()

    def cell_color(val):
        if pd.isna(val):
            return ''
        t = (val - min_r) / (max_r - min_r) if max_r > min_r else 0.5
        if t <= 0.5:
            frac = t * 2
            r = int(163 + (255 - 163) * frac)
            g = int(244 + (255 - 244) * frac)
            b = int(176 + (255 - 176) * frac)
        else:
            frac = (t - 0.5) * 2
            r = 255
            g = int(255 + (163 - 255) * frac)
            b = int(255 + (163 - 255) * frac)
        return f'background-color: rgb({r},{g},{b})'

    return df.map(cell_color)


def render_table(title: str, df: pd.DataFrame,
                 fmt: str = '{:.2f}',
                 is_ranking: bool = False,
                 is_straddle: bool = False):
    st.markdown(f'**{title}**')
    if is_ranking:
        styler = df.style.apply(lambda _: color_ranking(df), axis=None).format('{:.0f}', na_rep='')
    elif is_straddle:
        styler = df.style.format('{:,.0f}', na_rep='')
    else:
        styler = df.style.format(fmt, na_rep='')
    st.dataframe(styler, use_container_width=True)


if 'seed_offset' not in st.session_state:
    st.session_state.seed_offset = 0

st.title('Volatility Dashboard')

if st.button('🔄 Refresh'):
    st.session_state.seed_offset += 1

offset = st.session_state.seed_offset

bbg_spot = make_spot_vol(seed=0 + offset)
bbg_fwd = make_fwd_vol(seed=1 + offset)
bbg_ratio = make_ratio(bbg_spot, bbg_fwd)
bbg_rank = make_ranking(bbg_ratio)

marx_spot = make_spot_vol(seed=2 + offset)
marx_fwd = make_fwd_vol(seed=3 + offset)
marx_ratio = make_ratio(marx_spot, marx_fwd)
marx_rank = make_ranking(marx_ratio)

straddle_raw = np.array([
    [310, 507, 774],
    [437, 731, 1135],
    [529, 904, 1409],
    [670, 1168, 1822],
    [782, 1369, 2137],
    [919, 1599, 2506],
    [1088, 1892, 2997],
    [1218, 2142, 3444],
], dtype=float)
marx_straddle = apply_mask(pd.DataFrame(straddle_raw, index=EXPIRIES, columns=TENORS))

st.subheader('BBG')
c1, c2, c3, c4, _ = st.columns(5)
with c1:
    render_table('BBG Spot Volatility', bbg_spot)
with c2:
    render_table('BBG Forward Volatility', bbg_fwd)
with c3:
    render_table('BBG RollUp Vol Ratio', bbg_ratio)
with c4:
    render_table('BBG RollUp Ranking', bbg_rank, is_ranking=True)

st.subheader('Marx')
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    render_table('Marx Spot Volatility', marx_spot)
with c2:
    render_table('Marx Forward Volatility', marx_fwd)
with c3:
    render_table('Marx RollUp Vol Ratio', marx_ratio)
with c4:
    render_table('Marx RollUp Ranking', marx_rank, is_ranking=True)
with c5:
    render_table('Marx ATM Straddle Price', marx_straddle, is_straddle=True)
