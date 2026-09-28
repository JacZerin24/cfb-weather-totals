from __future__ import annotations
import itertools, math
from functools import lru_cache
import pandas as pd, numpy as np

from .cfbd_client import CFBDClient
from .deep_research import prep
from .model_bakeoff import feature_lists, prep_features, reg_models
from .predict_week import normalize_games, merge_prior_team_features, add_live_categories
from .utils import read_df, write_df, ensure_dir

GAME_ID=401856707
OLD_PATH='outputs/prospective/2026/snapshots/board_20260928T181357Z_run36463362628_a1_ffd2c669b36a.csv'
FIELDS=['closing_total','temperature_f','dewpoint_f','humidity','precipitation']

def one_row(path):
    df=read_df(path)
    return df[pd.to_numeric(df['game_id'],errors='coerce').eq(GAME_ID)].iloc[0]

def build_base(meta, src):
    r=meta.copy()
    for c in ['closing_total','line_provider','game_indoors','temperature_f','dewpoint_f','humidity','wind_mph','precipitation','snowfall',
              'home_conference','away_conference','home_classification','away_classification']:
        if c in src.index: r[c]=src[c]
    r['pressure']=np.nan
    return pd.DataFrame([r])

def main():
    hist=prep(read_df('data/processed/modeling_dataset.csv'))
    nums,cats=feature_lists(hist)
    hist=prep_features(hist,cats)
    model=reg_models(nums,cats)['hist_gradient_boosting']
    model.fit(hist[nums+cats],hist['market_residual'])

    client=CFBDClient()
    games=normalize_games(client.get('/games',{'id':GAME_ID}),2026)
    meta=games.iloc[0].copy()
    old=one_row(OLD_PATH)
    new=one_row('outputs/weekly_board.csv')
    old_df=merge_prior_team_features(build_base(meta,old))
    new_df=merge_prior_team_features(build_base(meta,new))

    def score(df):
        z=add_live_categories(df.copy())
        for c in nums:
            if c not in z.columns: z[c]=np.nan
            z[c]=pd.to_numeric(z[c],errors='coerce')
        for c in cats:
            if c not in z.columns: z[c]='missing'
            z[c]=z[c].astype(str).fillna('missing')
        return float(model.predict(z[nums+cats])[0])

    old_score=score(old_df); new_score=score(new_df)

    # exact Shapley attribution of old->new across changed fields
    oldvals={f:old_df.iloc[0].get(f,np.nan) for f in FIELDS}
    newvals={f:new_df.iloc[0].get(f,np.nan) for f in FIELDS}
    n=len(FIELDS)
    cache={}
    def coalition(S):
        key=tuple(sorted(S))
        if key in cache: return cache[key]
        d=old_df.copy()
        for f in S: d.loc[:,f]=newvals[f]
        cache[key]=score(d)
        return cache[key]
    phi={f:0.0 for f in FIELDS}
    facts=math.factorial
    allset=set(FIELDS)
    for f in FIELDS:
        others=[x for x in FIELDS if x!=f]
        for k in range(len(others)+1):
            for comb in itertools.combinations(others,k):
                S=set(comb)
                w=facts(k)*facts(n-k-1)/facts(n)
                phi[f]+=w*(coalition(S|{f})-coalition(S))

    rows=[]
    for f in FIELDS:
        # direct old-state one-at-a-time and new-state reversion, useful for thresholds
        d=old_df.copy(); d.loc[:,f]=newvals[f]
        old_one=score(d)
        d2=new_df.copy(); d2.loc[:,f]=oldvals[f]
        new_revert=score(d2)
        rows.append({
            'feature':f,'old_value':oldvals[f],'new_value':newvals[f],
            'shapley_residual_delta':phi[f],
            'old_state_one_at_a_time_delta':old_one-old_score,
            'new_state_revert_effect':new_score-new_revert,
        })
    detail=pd.DataFrame(rows).sort_values('shapley_residual_delta')
    write_df(detail,'outputs/alabama_msu_delta_attribution.csv')

    summary=pd.DataFrame([{
        'old_replayed_residual':old_score,
        'old_recorded_residual':float(old['pred_market_residual']),
        'new_replayed_residual':new_score,
        'new_recorded_residual':float(new['pred_market_residual']),
        'residual_change':new_score-old_score,
        'shapley_sum':sum(phi.values()),
        'old_market_total':float(old['closing_total']),
        'new_market_total':float(new['closing_total']),
        'old_projected_total':float(old['closing_total'])+old_score,
        'new_projected_total':float(new['closing_total'])+new_score,
        'home_pregame_elo':meta.get('home_pregame_elo'),
        'away_pregame_elo':meta.get('away_pregame_elo'),
        'neutral_site':meta.get('neutral_site'),
        'conference_game':meta.get('conference_game'),
    }])
    write_df(summary,'outputs/alabama_msu_delta_summary.csv')
    print(summary.to_string(index=False))
    print(detail.to_string(index=False))

if __name__=='__main__': main()
