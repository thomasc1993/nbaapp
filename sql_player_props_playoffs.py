# sql_player_props_playoffs.py
"""
Pure SQL needed by LightGBMPlayoffModel.retrieve_data()

We keep **two** strings:
    * VIEW_SETUP_SQL – every CREATE VIEW / helper DDL statement
    * PLAYER_GAMES_SQL – the one giant SELECT that actually returns rows
Nothing else – no Python, no logging.
"""

# ------------------------------------------------------------
# 1)  all CREATE VIEW statements (exactly as in the old file)
# ------------------------------------------------------------
VIEW_SETUP_SQL = """


    /* ────────────────────────────────────────────────────────────────
   Regular-season Synergy play-type portfolio – one row per
   (player, season).  All columns kept verbatim for flexible use.
   ──────────────────────────────────────────────────────────────── */
    CREATE TEMP VIEW IF NOT EXISTS player_season_playtypes_rs AS
    SELECT
        *
    FROM   player_season_playtypes
    WHERE  season_type = 'Regular Season';   -- ← the shift


 /* Regular-season per-game tracking numbers of the *current* season
    for every player.  The view lives only for this connection. */
    CREATE TEMP VIEW IF NOT EXISTS player_tracking_rs_for_join AS
        SELECT
        t.player_id, t.season, t.catch_shoot_fga AS rs_cs_fga,  t.catch_shoot_fg3a AS rs_cs_fg3a,
        t.catch_shoot_fg3_pct AS rs_cs_fg3_pct, t.catch_shoot_efg_pct AS rs_cs_efg_pct, t.pull_up_fga AS rs_pu_fga,
        t.pull_up_efg_pct AS rs_pu_efg_pct, t.drives AS rs_drives, t.drive_fga AS rs_drive_fga, 
        t.drive_fga AS rs_drive_fga, t.drive_pts_pct AS rs_drive_pts_pct, t.drive_fg_pct AS rs_drive_fg_pct, 
        t.drive_ast_pct AS rs_drive_ast_pct, t.drive_pf AS rs_drive_pf, t.paint_touches AS rs_paint_touches,
        t.paint_touch_fg_pct AS rs_paint_touch_fg_pct, t.paint_touch_pts_pct AS rs_paint_touch_pts_pct, 
        t.paint_touch_passes_pct AS rs_paint_touch_passes_pct, t.paint_touch_pf AS rs_paint_touch_pf,
        t.post_touches AS rs_post_touches, t.post_touch_fg_pct AS rs_post_touch_fg_pct, 
        t.post_touch_pts_pct AS rs_post_touch_pts_pct, t.post_touch_ast_pct AS rs_post_touch_ast_pct, 
        s.ab_the_break_fg_pct AS rs_ab_fg_pct, s.ab_the_break_fga AS rs_ab_fga, s.ab_the_break_fgm AS rs_ab_fgm, 
        s.bc_fg_pct AS rs_bc_fg_pct, s.bc_fga AS rs_bc_fga, s.bc_fgm AS rs_bc_fgm, s.c3_fg_pct AS rs_c3_fg_pct, 
        s.c3_fga AS rs_c3_fga, s.c3_fgm AS rs_c3_fgm, s.in_the_paint_fg_pct AS rs_paint_fg_pct, 
        s.in_the_paint_fga AS rs_paint_fga, s.in_the_paint_fgm AS rs_paint_fgm, s.lc3_fg_pct AS rs_lc3_fg_pct, 
        s.lc3_fga AS rs_lc3_fga, s.lc3_fgm AS rs_lc3_fgm, s.mr_fg_pct AS rs_mr_fg_pct, s.mr_fga AS rs_mr_fga,
        s.mr_fgm AS rs_mr_fgm, s.ra_fg_pct AS rs_ra_fg_pct, s.ra_fga AS rs_ra_fga,
        s.ra_fgm AS rs_ra_fgm, s.rc3_fg_pct AS rs_rc3_fg_pct, s.rc3_fga AS rs_rc3_fga,
        s.rc3_fgm AS rs_rc3_fgm

        FROM   player_tracking_stats  t
        LEFT   JOIN player_shot_locations s ON  s.player_id = t.player_id AND s.season    = t.season AND s.season_type = 'Regular Season'
        WHERE  t.season_type = 'Regular Season';


    CREATE TEMP VIEW IF NOT EXISTS team_defense_tracking_rs_for_join AS
        SELECT
        team_id, season, ov_freq AS rs_tdt_ov_freq,
        ov_pct_plusminus AS rs_tdt_ov_pct_plusminus, d_fga AS rs_tdt_d_fga, d_fgm AS rs_tdt_d_fgm,
        d_fg_pct AS rs_tdt_d_fg_pct, normal_fg_pct AS rs_tdt_normal_fg_pct, "2p_freq" AS rs_tdt_2p_freq,
        "2p_plusminus" AS rs_tdt_2p_plusminus, fg2a AS rs_tdt_fg2a, fg2m AS rs_tdt_fg2m,
        fg2_pct AS rs_tdt_fg2_pct, ns_fg2_pct AS rs_tdt_ns_fg2_pct, "3p_freq" AS rs_tdt_3p_freq,
        "3p_plusminus" AS rs_tdt_3p_plusminus, fg3a AS rs_tdt_fg3a, fg3m AS rs_tdt_fg3m,
        fg3_pct AS rs_tdt_fg3_pct, ns_fg3_pct AS rs_tdt_ns_fg3_pct, lt6f_freq AS rs_tdt_lt6f_freq,
        lt6f_plusminus AS rs_tdt_lt6f_plusminus, fga_lt_06 AS rs_tdt_fga_lt_06, fgm_lt_06 AS rs_tdt_fgm_lt_06,
        lt_06_pct AS rs_tdt_lt_06_pct, ns_lt_06_pct AS rs_tdt_ns_lt_06_pct, lt10f_freq AS rs_tdt_lt10f_freq,
        lt10f_plusminus AS rs_tdt_lt10f_plusminus, fga_lt_10 AS rs_tdt_fga_lt_10, fgm_lt_10 AS rs_tdt_fgm_lt_10,
        lt_10_pct AS rs_tdt_lt_10_pct, ns_lt_10_pct AS rs_tdt_ns_lt_10_pct, gt15f_freq AS rs_tdt_gt15f_freq,
        gt15f_plusminus AS rs_tdt_gt15f_plusminus, fga_gt_15 AS rs_tdt_fga_gt_15, fgm_gt_15 AS rs_tdt_fgm_gt_15,
        gt_15_pct AS rs_tdt_gt_15_pct, ns_gt_15_pct AS rs_tdt_ns_gt_15_pct
        FROM   team_defense_tracking
        WHERE  season_type = 'Regular Season';

     CREATE TEMP VIEW IF NOT EXISTS vegas_odds_for_join AS
        SELECT
            game_id,
            team_id,
            moneyline_odds,
            spread_line,
            total_line,
            over_odds,
            under_odds
        FROM   historical_nba_odds;

    CREATE TEMP VIEW IF NOT EXISTS player_on_court_rs_for_join AS
        SELECT
            vs_player_id        AS player_id,
            team_id,
            season,
            -- metric columns -------------------------------------------------------
           pace AS rs_on_on_pace, poss AS rs_on_on_poss, net_rating AS rs_on_on_net_rating, off_rating AS rs_on_on_off_rating,
            def_rating AS rs_on_on_def_rating, ts_pct AS rs_on_on_ts_pct, efg_pct AS rs_on_on_efg_pct, pie AS rs_on_on_pie,
            tm_tov_pct AS rs_on_on_tm_tov_pct, pts AS rs_on_on_pts, ast_pct AS rs_on_on_ast_pct, fga AS rs_on_on_fga,
            fg3a AS rs_on_on_fg3a, fta AS rs_on_on_fta, fgm AS rs_on_on_fgm, fg3m AS rs_on_on_fg3m,
            ftm AS rs_on_on_ftm, plus_minus AS rs_on_on_plus_minus, ast_ratio AS rs_on_on_ast_ratio, ast_to AS rs_on_on_ast_to,
            pf AS rs_on_on_pf, pfd AS rs_on_on_pfd, oreb AS rs_on_on_oreb, dreb AS rs_on_on_dreb,
            reb AS rs_on_on_reb


        FROM player_team_on_off_court
        WHERE season_type = 'Regular Season'
        AND court_status = 'On'
        AND (group_set IS NULL OR group_set = 'On/Off Court');


    CREATE TEMP VIEW IF NOT EXISTS player_off_court_rs_for_join AS
            SELECT
            vs_player_id        AS player_id,
            team_id,
            season,
            -- metric columns -------------------------------------------------------
           pace AS rs_on_off_pace, poss AS rs_on_off_poss, net_rating AS rs_on_off_net_rating, off_rating AS rs_on_off_off_rating,
            def_rating AS rs_on_off_def_rating, ts_pct AS rs_on_off_ts_pct, efg_pct AS rs_on_off_efg_pct, pie AS rs_on_off_pie,
            tm_tov_pct AS rs_on_off_tm_tov_pct, pts AS rs_on_off_pts, ast_pct AS rs_on_off_ast_pct, fga AS rs_on_off_fga,
            fg3a AS rs_on_off_fg3a, fta AS rs_on_off_fta, fgm AS rs_on_off_fgm, fg3m AS rs_on_off_fg3m,
            ftm AS rs_on_off_ftm, plus_minus AS rs_on_off_plus_minus, ast_ratio AS rs_on_off_ast_ratio, ast_to AS rs_on_off_ast_to,
            pf AS rs_on_off_pf, pfd AS rs_on_off_pfd, oreb AS rs_on_off_oreb, dreb AS rs_on_off_dreb,
            reb AS rs_on_off_reb


        FROM player_team_on_off_court
        WHERE season_type = 'Regular Season'
        AND court_status = 'Off'
        AND (group_set IS NULL OR group_set = 'On/Off Court');
"""

# ------------------------------------------------------------
# 2) the giant SELECT that pandas will run
# ------------------------------------------------------------
PLAYER_GAMES_SQL = """

    WITH opponent_position_onoff_rs AS (
         WITH opponent_players AS (
             SELECT
                 tgf.game_id,
                 tgf.team_id,
                 tgf.opponent_team_id,
                 tgf.season,
                 ps.position_number,
                 pgf.player_id,
                 pgf.min                    -- so you can weight if you wish
             FROM   teams_game_features tgf
             JOIN   player_game_features pgf
                    ON pgf.game_id = tgf.game_id
                   AND pgf.team_id = tgf.opponent_team_id
                   AND pgf.min     > 0      -- only guys who played
             JOIN   players ps
                    ON ps.player_id = pgf.player_id
         )
         SELECT
             op.game_id,
             op.team_id,
             op.position_number,
             /* ---------- season ON-court minutes-weighted averages ---------- */
            SUM(onv.rs_on_on_net_rating  * op.min) / NULLIF(SUM(op.min),0)  AS opp_pos_on_on_net_rating,
            SUM(onv.rs_on_on_off_rating  * op.min) / NULLIF(SUM(op.min),0)  AS opp_pos_on_on_off_rating,
            SUM(onv.rs_on_on_def_rating  * op.min) / NULLIF(SUM(op.min),0)  AS opp_pos_on_on_def_rating,
            SUM(onv.rs_on_on_pace        * op.min) / NULLIF(SUM(op.min),0)  AS opp_pos_on_on_pace,
            SUM(onv.rs_on_on_poss        * op.min) / NULLIF(SUM(op.min),0)  AS opp_pos_on_on_poss,
            SUM(onv.rs_on_on_dreb        * op.min) / NULLIF(SUM(op.min),0)  AS opp_pos_on_on_dreb,
            SUM(onv.rs_on_on_pie         * op.min) / NULLIF(SUM(op.min),0)  AS opp_pos_on_on_pie,
            SUM(onv.rs_on_on_pf          * op.min) / NULLIF(SUM(op.min),0)  AS opp_pos_on_on_pf,

            /* ---------- season OFF-court minutes-weighted averages --------- */
            SUM(offv.rs_on_off_net_rating * op.min) / NULLIF(SUM(op.min),0) AS opp_pos_on_off_net_rating,
            SUM(offv.rs_on_off_off_rating * op.min) / NULLIF(SUM(op.min),0) AS opp_pos_on_off_off_rating,
            SUM(offv.rs_on_off_def_rating * op.min) / NULLIF(SUM(op.min),0) AS opp_pos_on_off_def_rating,
            SUM(offv.rs_on_off_pace       * op.min) / NULLIF(SUM(op.min),0) AS opp_pos_on_off_pace,
            SUM(offv.rs_on_off_poss       * op.min) / NULLIF(SUM(op.min),0) AS opp_pos_on_off_poss,
            SUM(offv.rs_on_off_dreb       * op.min) / NULLIF(SUM(op.min),0) AS opp_pos_on_off_dreb,
            SUM(offv.rs_on_off_pie        * op.min) / NULLIF(SUM(op.min),0) AS opp_pos_on_off_pie,
            SUM(offv.rs_on_off_pf         * op.min) / NULLIF(SUM(op.min),0) AS opp_pos_on_off_pf

         FROM         opponent_players             op
         LEFT JOIN    player_on_court_rs_for_join  onv
                ON onv.player_id = op.player_id
               AND onv.team_id   = op.opponent_team_id
               AND onv.season    = op.season
         LEFT JOIN    player_off_court_rs_for_join offv
                ON offv.player_id = op.player_id
               AND offv.team_id   = op.opponent_team_id
               AND offv.season    = op.season
         GROUP BY
             op.game_id,
             op.team_id,
             op.position_number
    ), 

    pgf_with_avgs AS (
        SELECT
            pgf.*,

            /* ---------- career average up to yesterday ---------- */
            AVG(pgf.pts) OVER (
                PARTITION BY pgf.player_id
                ORDER BY     pgf.game_date, pgf.game_id          -- tie-break
                ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
            ) AS pts_career_avg,

            /* ---------- season average up to yesterday ---------- */
            AVG(pgf.pts) OVER (
                PARTITION BY pgf.player_id, pgf.season
                ORDER BY     pgf.game_date, pgf.game_id
                ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
            ) AS pts_season_avg
        FROM player_game_features pgf
    ),
    
        /* =============================================================
        1. Identify season stars, primary team, and team-level ranks
        =============================================================*/
        player_stats AS (               -- composite rankings per player/season
            SELECT
                season,
                player_id,
                composite_rank
            FROM   player_historical_composite_stats
        ),
        /* -------------------------------------------------------------
        Top-45 league-wide “stars” each season (lower rank = better)
        -------------------------------------------------------------*/
        star_players AS (
            SELECT
                season,
                player_id,
                1 AS is_star
            FROM (
                SELECT
                    season,
                    player_id,
                    composite_rank,
                    RANK() OVER (PARTITION BY season
                                ORDER BY composite_rank) AS season_rank
                FROM   player_stats
            )
            WHERE  season_rank <= 45           -- ← top-45 per season
        ),
        /* -------------------------------------------------------------
        Determine a player’s main team in the given season
        -------------------------------------------------------------*/
        player_team_games AS (                -- games per (season, player, team)
            SELECT
                season,
                player_id,
                team_id,
                COUNT(*) AS gms
            FROM   player_game_features
            GROUP  BY season, player_id, team_id
        ),
        main_team AS (                        -- keep the row with most games
            SELECT *
            FROM (
                SELECT *,
                    ROW_NUMBER() OVER (PARTITION BY season, player_id
                                        ORDER BY gms DESC, team_id) AS rn
                FROM   player_team_games
            )
            WHERE rn = 1
        ),
        /* -------------------------------------------------------------
        Rank each team’s stars by composite_rank (best = 1)
        -------------------------------------------------------------*/
        team_star_rank AS (
            SELECT *
            FROM (
                SELECT
                    ps.season,
                    mt.team_id,
                    ps.player_id,
                    ps.composite_rank,
                    RANK() OVER (PARTITION BY ps.season, mt.team_id
                                ORDER BY ps.composite_rank) AS star_rank
                FROM   player_stats      ps
                JOIN   star_players      sp ON  sp.season    = ps.season
                                        AND sp.player_id = ps.player_id
                JOIN   main_team         mt ON  mt.season    = ps.season
                                        AND mt.player_id = ps.player_id
            )
            WHERE star_rank <= 2               -- keep top-2 stars per team
        )
        /* =============================================================
        2. For each game/team, flag whether Star-1 or Star-2 is out
        =============================================================*/
        , star_out_flags AS (
            SELECT
                g.game_id,
                g.team_id,
                MAX(CASE WHEN tsr.star_rank = 1 THEN 1 ELSE 0 END) AS star1_out_flag,
                MAX(CASE WHEN tsr.star_rank = 2 THEN 1 ELSE 0 END) AS star2_out_flag
            FROM   teams_game_features   g

            LEFT   JOIN player_game_features inj
                ON inj.game_id   = g.game_id
                AND inj.team_id   = g.team_id
                AND inj.available_flag = 0
            /* Tie the injury row to its team’s stars */
            LEFT   JOIN team_star_rank   tsr
                ON tsr.season    = inj.season
                AND tsr.player_id = inj.player_id
                AND tsr.team_id   = g.team_id
            GROUP  BY g.game_id, g.team_id
        )


    SELECT
    /* Player Row */
    pgf.*, 
    ps.position_number AS player_position_number,
    ps.greatest_75_flag AS player_greatest_75,
    ps.draft_round AS player_draft_round,
    ps.draft_number AS player_draft_number,
    ps.height_mm AS player_height,
    ps.weight AS player_weight,

    /* Team Context */
    tgf.win_streak AS team_win_streak, tgf.losing_streak AS team_losing_streak,
    tgf.road_trip_length AS team_road_trip_length, tgf.travel_distance AS team_travel_distance,
    tgf.is_back_to_back AS team_is_back_to_back, tgf.is_three_in_four AS team_is_three_in_four,
    tgf.is_five_in_seven AS team_is_five_in_seven, tgf.is_high_altitude AS team_is_high_altitude,
    tgf.is_home AS is_home,          -- the name the FE code expects
    tgf.is_home AS team_is_home,     -- keep this alias if other code still uses it
    tgf.wl AS team_wl, /* Select wl column for win derivation */
    tgf.min AS team_min, tgf.pace AS team_pace, tgf.e_pace AS team_e_pace, tgf.poss AS team_poss,
    tgf.pie AS team_pie, tgf.off_rating AS team_off_rating, tgf.e_off_rating AS team_e_off_rating,
    tgf.def_rating AS team_def_rating, tgf.net_rating AS team_net_rating, tgf.plus_minus AS team_plus_minus,
    tgf.pts AS team_pts, tgf.ast AS team_ast, tgf.ast_ratio AS team_ast_ratio,
    tgf.tm_tov_pct AS team_tm_tov_pct, tgf.efg_pct AS team_efg_pct, tgf.ts_pct AS team_ts_pct,
    tgf.pct_pts_3pt AS team_pct_pts_3pt, tgf.pct_pts_2pt AS team_pct_pts_2pt, tgf.pf AS team_pf,
    tgf.pct_pts_2pt_mr AS team_pct_pts_2pt_mr, tgf.pct_pts_ft AS team_pct_pts_ft,
    tgf.pct_ast_2pm AS team_pct_ast_2pm, tgf.pct_uast_2pm AS team_pct_uast_2pm,
    tgf.pct_ast_3pm AS team_pct_ast_3pm, tgf.pct_uast_3pm AS team_pct_uast_3pm,
    tgf.is_playoffs AS is_playoffs, tgf.playoff_round AS playoff_round,
    tgf.series_number AS series_number, tgf.series_record AS series_record,
    tgf.is_elimination_game AS is_elimination_game, tgf.can_win_series AS can_win_series,
    tgf.is_game_6 AS is_game_6, tgf.is_game_7 AS is_game_7, tgf.has_home_court AS has_home_court,
    tgf.series_score_diff AS series_score_diff, tgf.series_prev_game_margin AS series_prev_game_margin,

    /* Opponent Context */
    otgf.win_streak AS opponent_win_streak, otgf.losing_streak AS opponent_losing_streak, 
    otgf.is_back_to_back AS opponent_is_back_to_back, otgf.is_three_in_four AS opponent_is_three_in_four, 
    otgf.pace AS opponent_pace, otgf.off_rating AS opponent_off_rating,
    otgf.def_rating AS opponent_def_rating, otgf.net_rating AS opponent_net_rating, otgf.dreb_pct AS opponent_dreb_pct,
    otgf.pf AS opponent_pf, otgf.tm_tov_pct AS opponent_tm_tov_pct, otgf.efg_pct AS opponent_efg_pct,
    otgf.ts_pct AS opponent_ts_pct,

    /* ────────── Player Regular Season Tracking Metric Averages ────────── */
    rts.rs_cs_fg3a AS rs_cs_fg3a, rts.rs_cs_fga AS rs_cs_fga, rts.rs_cs_fg3_pct AS rs_cs_fg3_pct, rts.rs_cs_efg_pct AS rs_cs_efg_pct,
    rts.rs_pu_fga AS rs_pu_fga, rts.rs_pu_efg_pct AS rs_pu_efg_pct, rts.rs_drives AS rs_drives,
    rts.rs_drive_pts_pct AS rs_drive_pts_pct, rts.rs_drive_fg_pct AS rs_drive_fg_pct, rts.rs_drive_ast_pct AS rs_drive_ast_pct,
    rts.rs_drive_fga AS rs_drive_fga, rts.rs_drive_pf AS rs_drive_pf, rts.rs_paint_touch_pf AS rs_paint_touch_pf,
    rts.rs_paint_touches AS rs_paint_touches, rts.rs_paint_touch_fg_pct AS rs_paint_touch_fg_pct, rts.rs_paint_touch_pts_pct AS rs_paint_touch_pts_pct,
    rts.rs_paint_touch_passes_pct AS rs_paint_touch_passes_pct, rts.rs_post_touches AS rs_post_touches, rts.rs_post_touch_fg_pct AS rs_post_touch_fg_pct,
    rts.rs_post_touch_pts_pct AS rs_post_touch_pts_pct, rts.rs_post_touch_ast_pct AS rs_post_touch_ast_pct, rts.rs_ab_fg_pct AS rs_ab_fg_pct,
    rts.rs_ab_fga AS rs_ab_fga, rts.rs_ab_fgm AS rs_ab_fgm, rts.rs_bc_fg_pct AS rs_bc_fg_pct,
    rts.rs_bc_fga AS rs_bc_fga, rts.rs_bc_fgm AS rs_bc_fgm, rts.rs_c3_fg_pct AS rs_c3_fg_pct,
    rts.rs_c3_fga AS rs_c3_fga, rts.rs_c3_fgm AS rs_c3_fgm, rts.rs_paint_fg_pct AS rs_paint_fg_pct,
    rts.rs_paint_fga AS rs_paint_fga, rts.rs_paint_fgm AS rs_paint_fgm, rts.rs_lc3_fg_pct AS rs_lc3_fg_pct,
    rts.rs_lc3_fga AS rs_lc3_fga, rts.rs_lc3_fgm AS rs_lc3_fgm, rts.rs_mr_fg_pct AS rs_mr_fg_pct,
    rts.rs_mr_fga AS rs_mr_fga, rts.rs_mr_fgm AS rs_mr_fgm, rts.rs_ra_fg_pct AS rs_ra_fg_pct,
    rts.rs_ra_fga AS rs_ra_fga, rts.rs_ra_fgm AS rs_ra_fgm, rts.rs_rc3_fg_pct AS rs_rc3_fg_pct,
    rts.rs_rc3_fga AS rs_rc3_fga, rts.rs_rc3_fgm AS rs_rc3_fgm,


    /* ────────── Opponent Team Regular Season Defense Tracking Metric Averages ────────── */
    tdt.rs_tdt_d_fg_pct AS opp_rs_d_fg_pct, tdt.rs_tdt_d_fga AS opp_rs_d_fga, tdt.rs_tdt_d_fgm AS opp_rs_d_fgm,
    tdt.rs_tdt_ov_pct_plusminus AS opp_rs_ov_pct_plusminus, tdt.rs_tdt_2p_freq AS opp_rs_2p_freq, tdt.rs_tdt_2p_plusminus AS opp_rs_2p_plusminus,
    tdt.rs_tdt_3p_freq AS opp_rs_3p_freq, tdt.rs_tdt_3p_plusminus AS opp_rs_3p_plusminus, tdt.rs_tdt_lt6f_freq AS opp_rs_lt6f_freq,
    tdt.rs_tdt_lt6f_plusminus AS opp_rs_lt6f_plusminus, tdt.rs_tdt_lt10f_freq AS opp_rs_lt10f_freq, tdt.rs_tdt_lt10f_plusminus AS opp_rs_lt10f_plusminus,
    tdt.rs_tdt_gt15f_freq AS opp_rs_gt15f_freq, tdt.rs_tdt_gt15f_plusminus AS opp_rs_gt15f_plusminus, tdt.rs_tdt_lt_06_pct AS opp_rs_lt_06_pct,
    tdt.rs_tdt_lt_10_pct AS opp_rs_lt_10_pct, tdt.rs_tdt_gt_15_pct AS opp_rs_gt_15_pct, tdt.rs_tdt_fg2_pct AS opp_rs_fg2_pct,
    tdt.rs_tdt_fg3_pct AS opp_rs_fg3_pct,

    /* ────────── Vegas Odds ────────── */
    vo.moneyline_odds                                            AS moneyline_odds,
    CASE WHEN vo.moneyline_odds IS NOT NULL THEN 1 ELSE 0 END    AS has_moneyline_odds,

    vo.spread_line                                               AS spread_line,
    CASE WHEN vo.spread_line  IS NOT NULL THEN 1 ELSE 0 END      AS has_spread_line,

    vo.total_line                                                AS total_line,
    CASE WHEN vo.total_line   IS NOT NULL THEN 1 ELSE 0 END      AS has_total_line,

    vo.over_odds                                                 AS over_odds,
    CASE WHEN vo.over_odds    IS NOT NULL THEN 1 ELSE 0 END      AS has_over_odds,

    vo.under_odds                                                AS under_odds,
    CASE WHEN vo.under_odds  IS NOT NULL THEN 1 ELSE 0 END       AS has_under_odds,

    /* ────────── Player Regular-Season ON-COURT metrics ────────── */
    pon.rs_on_on_pace AS rs_on_on_pace, pon.rs_on_on_poss AS rs_on_on_poss, pon.rs_on_on_net_rating AS rs_on_on_net_rating, pon.rs_on_on_off_rating AS rs_on_on_off_rating,
    pon.rs_on_on_def_rating AS rs_on_on_def_rating, pon.rs_on_on_ts_pct AS rs_on_on_ts_pct, pon.rs_on_on_efg_pct AS rs_on_on_efg_pct, pon.rs_on_on_pie AS rs_on_on_pie,
    pon.rs_on_on_tm_tov_pct AS rs_on_on_tm_tov_pct, pon.rs_on_on_pts AS rs_on_on_pts, pon.rs_on_on_ast_pct AS rs_on_on_ast_pct, pon.rs_on_on_fga AS rs_on_on_fga,
    pon.rs_on_on_fg3a AS rs_on_on_fg3a, pon.rs_on_on_fta AS rs_on_on_fta, pon.rs_on_on_fgm AS rs_on_on_fgm, pon.rs_on_on_fg3m AS rs_on_on_fg3m,
    pon.rs_on_on_ftm AS rs_on_on_ftm, pon.rs_on_on_plus_minus AS rs_on_on_plus_minus, pon.rs_on_on_ast_ratio AS rs_on_on_ast_ratio, pon.rs_on_on_ast_to AS rs_on_on_ast_to,
    pon.rs_on_on_pf AS rs_on_on_pf, pon.rs_on_on_pfd AS rs_on_on_pfd, pon.rs_on_on_oreb AS rs_on_on_oreb, pon.rs_on_on_dreb AS rs_on_on_dreb,
    pon.rs_on_on_reb AS rs_on_on_reb,


    /* ────────── Player Regular-Season OFF-COURT metrics ───────── */
    poff.rs_on_off_pace AS rs_on_off_pace, poff.rs_on_off_poss AS rs_on_off_poss, poff.rs_on_off_net_rating AS rs_on_off_net_rating, poff.rs_on_off_off_rating AS rs_on_off_off_rating,
    poff.rs_on_off_def_rating AS rs_on_off_def_rating, poff.rs_on_off_ts_pct AS rs_on_off_ts_pct, poff.rs_on_off_efg_pct AS rs_on_off_efg_pct, poff.rs_on_off_pie AS rs_on_off_pie,
    poff.rs_on_off_tm_tov_pct AS rs_on_off_tm_tov_pct, poff.rs_on_off_pts AS rs_on_off_pts, poff.rs_on_off_ast_pct AS rs_on_off_ast_pct, poff.rs_on_off_fga AS rs_on_off_fga,
    poff.rs_on_off_fg3a AS rs_on_off_fg3a, poff.rs_on_off_fta AS rs_on_off_fta, poff.rs_on_off_fgm AS rs_on_off_fgm, poff.rs_on_off_fg3m AS rs_on_off_fg3m,
    poff.rs_on_off_ftm AS rs_on_off_ftm, poff.rs_on_off_plus_minus AS rs_on_off_plus_minus, poff.rs_on_off_ast_ratio AS rs_on_off_ast_ratio, poff.rs_on_off_ast_to AS rs_on_off_ast_to,
    poff.rs_on_off_pf AS rs_on_off_pf, poff.rs_on_off_pfd AS rs_on_off_pfd, poff.rs_on_off_oreb AS rs_on_off_oreb, poff.rs_on_off_dreb AS rs_on_off_dreb,
    poff.rs_on_off_reb AS rs_on_off_reb,

    /* ────────── Handy ON-minus-OFF deltas (explicit signal) ───── */
    (pon.rs_on_on_pace - poff.rs_on_off_pace) AS rs_onoff_pace_diff, (pon.rs_on_on_poss - poff.rs_on_off_poss) AS rs_onoff_poss_diff, (pon.rs_on_on_net_rating - poff.rs_on_off_net_rating) AS rs_onoff_net_rating_diff, (pon.rs_on_on_off_rating - poff.rs_on_off_off_rating) AS rs_onoff_off_rating_diff,
    (pon.rs_on_on_def_rating - poff.rs_on_off_def_rating) AS rs_onoff_def_rating_diff, (pon.rs_on_on_ts_pct - poff.rs_on_off_ts_pct) AS rs_onoff_ts_pct_diff, (pon.rs_on_on_efg_pct - poff.rs_on_off_efg_pct) AS rs_onoff_efg_pct_diff, (pon.rs_on_on_pie - poff.rs_on_off_pie) AS rs_onoff_pie_diff,
    (pon.rs_on_on_tm_tov_pct - poff.rs_on_off_tm_tov_pct) AS rs_onoff_tm_tov_pct_diff, (pon.rs_on_on_pts - poff.rs_on_off_pts) AS rs_onoff_pts_diff, (pon.rs_on_on_ast_pct - poff.rs_on_off_ast_pct) AS rs_onoff_ast_pct_diff, (pon.rs_on_on_fga - poff.rs_on_off_fga) AS rs_onoff_fga_diff,
    (pon.rs_on_on_fg3a - poff.rs_on_off_fg3a) AS rs_onoff_fg3a_diff, (pon.rs_on_on_fta - poff.rs_on_off_fta) AS rs_onoff_fta_diff, (pon.rs_on_on_fgm - poff.rs_on_off_fgm) AS rs_onoff_fgm_diff, (pon.rs_on_on_fg3m - poff.rs_on_off_fg3m) AS rs_onoff_fg3m_diff,
    (pon.rs_on_on_ftm - poff.rs_on_off_ftm) AS rs_onoff_ftm_diff, (pon.rs_on_on_plus_minus - poff.rs_on_off_plus_minus) AS rs_onoff_pm_diff, (pon.rs_on_on_ast_ratio - poff.rs_on_off_ast_ratio) AS rs_onoff_ast_ratio_diff, (pon.rs_on_on_ast_to - poff.rs_on_off_ast_to) AS rs_onoff_ast_to_diff,
    (pon.rs_on_on_pf - poff.rs_on_off_pf) AS rs_onoff_pf_diff, (pon.rs_on_on_pfd - poff.rs_on_off_pfd) AS rs_onoff_pfd_diff, (pon.rs_on_on_oreb - poff.rs_on_off_oreb) AS rs_onoff_oreb_diff, (pon.rs_on_on_dreb - poff.rs_on_off_dreb) AS rs_onoff_dreb_diff,
    (pon.rs_on_on_reb - poff.rs_on_off_reb) AS rs_onoff_reb_diff,

    /* ---------- opponent-position ON-court (minutes-weighted) ---------- */
    opo.opp_pos_on_on_net_rating   AS opp_pos_on_on_net_rating,
    opo.opp_pos_on_on_off_rating   AS opp_pos_on_on_off_rating,
    opo.opp_pos_on_on_def_rating   AS opp_pos_on_on_def_rating,
    opo.opp_pos_on_on_pace         AS opp_pos_on_on_pace,
    opo.opp_pos_on_on_poss         AS opp_pos_on_on_poss,
    opo.opp_pos_on_on_dreb         AS opp_pos_on_on_dreb,
    opo.opp_pos_on_on_pie          AS opp_pos_on_on_pie,
    opo.opp_pos_on_on_pf           AS opp_pos_on_on_pf,

    /* ---------- opponent-position OFF-court (minutes-weighted) --------- */
    opo.opp_pos_on_off_net_rating  AS opp_pos_on_off_net_rating,
    opo.opp_pos_on_off_off_rating  AS opp_pos_on_off_off_rating,
    opo.opp_pos_on_off_def_rating  AS opp_pos_on_off_def_rating,
    opo.opp_pos_on_off_pace        AS opp_pos_on_off_pace,
    opo.opp_pos_on_off_poss        AS opp_pos_on_off_poss,
    opo.opp_pos_on_off_dreb        AS opp_pos_on_off_dreb,
    opo.opp_pos_on_off_pie         AS opp_pos_on_off_pie,
    opo.opp_pos_on_off_pf          AS opp_pos_on_off_pf,

        /* Opponent vs Position */
    ovp.fgm_allowed AS opponent_vs_player_fgm_allowed, ovp.fga_allowed AS opponent_vs_player_fga_allowed,
    ovp.pts_allowed AS opponent_vs_player_pts_allowed,


    /* ────────── Player Advanced Game Tracking ───── */
    
    pgat.speed AS player_game_speed, pgat.dist AS player_game_dist, pgat.oreb_c AS player_game_oreb_c,
    pgat.dreb_c AS player_game_dreb_c, pgat.reb_c AS player_game_reb_c, pgat.touches AS player_game_touches,
    pgat.screen_ast AS player_game_screen_ast, pgat.ft_ast AS player_game_ft_ast, pgat.passes AS player_game_passes,
    pgat.ast AS player_game_ast, pgat.c_fgm AS player_game_c_fgm, pgat.c_fga AS player_game_c_fga,
    pgat.c_fg_pct AS player_game_c_fg_pct, pgat.u_fgm AS player_game_u_fgm, pgat.u_fga AS player_game_u_fga,
    pgat.u_fg_pct AS player_game_u_fg_pct, pgat.d_fgm AS player_game_d_fgm,
    pgat.d_fga AS player_game_d_fga, pgat.d_fg_pct AS player_game_d_fg_pct, pgat.min_numeric AS player_game_min_numeric,
    pgat.dist_per_min AS player_game_dist_per_min, pgat.touches_per_min AS player_game_touches_per_min, pgat.passes_per_min AS player_game_passes_per_min, pgat.ast_per_min AS player_game_ast_per_min,
    pgat.screen_ast_per_min AS player_game_screen_ast_per_min, pgat.ft_ast_per_min AS player_game_ft_ast_per_min, pgat.reb_c_per_min AS player_game_reb_c_per_min,
    pgat.oreb_c_ratio AS player_game_oreb_c_ratio, pgat.dreb_c_ratio AS player_game_dreb_c_ratio, pgat.passes_per_touch AS player_game_passes_per_touch,
    pgat.ast_per_touch AS player_game_ast_per_touch, pgat.high_value_touch_pct AS player_game_high_value_touch_pct, pgat.ast_conversion_pct AS player_game_ast_conversion_pct,
    pgat.total_fga AS player_game_total_fga, pgat.contest_share AS player_game_contest_share, pgat.uncontest_share AS player_game_uncontest_share,
    pgat.cfg_diff AS player_game_cfg_diff, pgat.weighted_fg_pct AS player_game_weighted_fg_pct, pgat.defensive_activity AS player_game_defensive_activity,
    pgat.defended_shot_share AS player_game_defended_shot_share, pgat.speed_x_min AS player_game_speed_x_min, pgat.dist_minus_expected AS player_game_dist_minus_expected,
    pgat.touches_x_speed AS player_game_touches_x_speed, pgat.passes_x_contest_share AS player_game_passes_x_contest_share,
    pgat.is_starter AS player_game_is_starter,


     /* ────────── Player per-75-possession rates ────────── */
    (100.0 * pgf.fga)  / NULLIF(pgf.poss, 0)  AS player_game_fga_per100,
    (100.0 * pgf.fgm)  / NULLIF(pgf.poss, 0)  AS player_game_fgm_per100,
    (100.0 * pgf.fta)  / NULLIF(pgf.poss, 0)  AS player_game_fta_per100,
    (100.0 * pgf.ftm)  / NULLIF(pgf.poss, 0)  AS player_game_ftm_per100,
    (100.0 * pgf.fg3a) / NULLIF(pgf.poss, 0)  AS player_game_fg3a_per100,
    (100.0 * pgf.fg3m) / NULLIF(pgf.poss, 0)  AS player_game_fg3m_per100,
    (100.0 * pgat.touches) / NULLIF(pgf.poss, 0)  AS player_game_touches_per100,
    (100.0 * pgat.passes) / NULLIF(pgf.poss, 0)  AS player_game_passes_per100,
    (100.0 * pgf.pts)  / NULLIF(pgf.poss, 0)  AS player_game_pts_per100,
    (100.0 * pgf.pfd)  / NULLIF(pgf.poss, 0)  AS player_game_pfd_per100,
    (100.0 * pgat.dist)  / NULLIF(pgf.poss, 0)  AS player_game_dist_per100,
    (100.0 * pgf.ast)  / NULLIF(pgf.poss, 0)  AS player_game_ast_per100,
    (100.0 * pgf.nba_fantasy_pts)  / NULLIF(pgf.poss, 0)  AS player_game_nba_fantasy_pts_per100,
    (100.0 * pgf.min)  / NULLIF(pgf.poss, 0)  AS player_game_min_per100,
    (100.0 * pgf.min)  / NULLIF(pgf.poss, 0)  AS player_game_min_per100,

    pgf.min / 48.0                              AS min_share_of_game,
    pgf.poss / NULLIF(tgf.poss,0)               AS poss_share_of_game,
    (48.0 * pgf.poss) / NULLIF(tgf.poss,0)      AS pace_adj_minutes_100,
    

    (100.0 * pgat.defended_shot_share)  / NULLIF(pgf.poss, 0)  AS player_game_defended_shot_share_per100,
    (100.0 * ovp.fgm_allowed)  / NULLIF(pgf.poss, 0)  AS player_game_opponent_vs_player_fgm_allowed_per100,
    (100.0 * ovp.pts_allowed)  / NULLIF(pgf.poss, 0)  AS player_game_opponent_vs_player_pts_allowed_per100,
    (100.0 * ovp.fgm_allowed)  / NULLIF(pgf.poss, 0)  AS player_game_opponent_vs_player_fga_allowed_per100,
    

    
    pgf.fga / NULLIF(pgat.touches, 0)  AS player_game_fga_per_touch,
    pgf.fga / NULLIF(pgf.poss, 0)  AS player_game_fga_per_poss,
    pgf.nba_fantasy_pts / NULLIF(pgf.poss, 0)  AS player_game_nba_fantasy_pts_per_poss,
    pgf.fta / NULLIF(pgf.poss, 0)  AS player_game_fta_per_poss,
    pgf.pfd / NULLIF(pgf.poss, 0)  AS player_game_pfd_per_poss,
    pgf.ast / NULLIF(pgf.poss, 0)  AS player_game_ast_per_poss,
    pgf.fg3a / NULLIF(pgf.poss, 0)  AS player_game_fg3a_per_poss,
    pgat.passes / NULLIF(pgf.poss, 0)  AS player_game_passes_per_poss,
    pgat.defended_shot_share / NULLIF(pgf.poss, 0)  AS player_game_defended_shot_share_per_poss,
    pgat.touches / NULLIF(pgf.poss, 0)  AS player_game_touches_per_poss,
    pgat.dist / NULLIF(pgf.poss, 0)  AS player_game_dist_per_poss,
    pgat.speed / NULLIF(pgf.poss, 0)  AS player_game_speed_per_poss,
    ovp.fga_allowed / NULLIF(pgf.poss, 0)  AS player_game_opponent_vs_player_fga_allowed_per_poss,
    ovp.fgm_allowed / NULLIF(pgf.poss, 0)  AS player_game_opponent_vs_player_fgm_allowed_per_poss,
    ovp.pts_allowed / NULLIF(pgf.poss, 0)  AS player_game_opponent_vs_player_pts_allowed_per_poss,

    COALESCE(sp.is_star, 0) AS player_is_star,
    COALESCE(sof.star1_out_flag,0)  AS star1_out_flag,
    COALESCE(sof.star2_out_flag,0)  AS star2_out_flag,
    

    CASE WHEN pgat.is_starter = 1 THEN pgf.min ELSE NULL END AS player_game_starter_min,
    CASE WHEN pgat.is_starter = 0 THEN pgf.min  ELSE NULL END AS player_game_bench_min,
    CASE WHEN pgat.is_starter = 1 THEN 0 ELSE 1 END AS player_game_is_bench,
    CASE WHEN pgf.is_available = 1 THEN 1 ELSE 0 END AS player_game_is_injury,
    CASE WHEN pgf.is_available = 0 THEN 1 ELSE 0 END AS player_game_is_available,


    
        /* ────────── NBA RAPM Player Regular Season Play Type Advanced Metrics ───── */


    pspt.handler_adj_percentile AS pt_handler_adj_percentile, pspt.handler_adj_poss AS pt_handler_adj_poss, pspt.handler_adj_ppp AS pt_handler_adj_ppp, pspt.handler_adj_pts AS pt_handler_adj_pts,
    pspt.handler_efg_pct AS pt_handler_efg_pct, pspt.handler_fga AS pt_handler_fga, pspt.handler_fga_per_g AS pt_handler_fga_per_g, pspt.handler_fga_percentile AS pt_handler_fga_percentile,
    pspt.handler_fgm AS pt_handler_fgm, pspt.handler_fgm_x AS pt_handler_fgm_x, pspt.handler_fg_pct AS pt_handler_fg_pct, pspt.handler_ft_poss_pct AS pt_handler_ft_poss_pct,
    pspt.handler_gp AS pt_handler_gp, pspt.handler_percentile AS pt_handler_percentile, pspt.handler_plusone_poss_pct AS pt_handler_plusone_poss_pct, pspt.handler_poe AS pt_handler_poe,
    pspt.handler_poe_per_g AS pt_handler_poe_per_g, pspt.handler_poe_rank AS pt_handler_poe_rank, pspt.handler_poss AS pt_handler_poss, pspt.handler_poss_per_g AS pt_handler_poss_per_g,
    pspt.handler_poss_percentile AS pt_handler_poss_percentile, pspt.handler_poss_weighted_ppp AS pt_handler_poss_weighted_ppp, pspt.handler_ppp AS pt_handler_ppp, pspt.handler_pts AS pt_handler_pts,
    pspt.handler_score_poss_pct AS pt_handler_score_poss_pct, pspt.handler_sf_poss_pct AS pt_handler_sf_poss_pct, pspt.handler_tov_poss_pct AS pt_handler_tov_poss_pct, pspt.handler_rppp AS pt_handler_rppp,
    pspt.iso_adj_percentile AS pt_iso_adj_percentile, pspt.iso_adj_poss AS pt_iso_adj_poss, pspt.iso_adj_ppp AS pt_iso_adj_ppp, pspt.iso_adj_pts AS pt_iso_adj_pts,
    pspt.iso_efg_pct AS pt_iso_efg_pct, pspt.iso_fga AS pt_iso_fga, pspt.iso_fga_per_g AS pt_iso_fga_per_g, pspt.iso_fga_percentile AS pt_iso_fga_percentile,
    pspt.iso_fgm AS pt_iso_fgm, pspt.iso_fgm_x AS pt_iso_fgm_x, pspt.iso_fg_pct AS pt_iso_fg_pct, pspt.iso_ft_poss_pct AS pt_iso_ft_poss_pct,
    pspt.iso_gp AS pt_iso_gp, pspt.iso_percentile AS pt_iso_percentile, pspt.iso_plusone_poss_pct AS pt_iso_plusone_poss_pct, pspt.iso_poe AS pt_iso_poe,
    pspt.iso_poe_per_g AS pt_iso_poe_per_g, pspt.iso_poe_rank AS pt_iso_poe_rank, pspt.iso_poss AS pt_iso_poss, pspt.iso_poss_per_g AS pt_iso_poss_per_g,
    pspt.iso_poss_percentile AS pt_iso_poss_percentile, pspt.iso_poss_weighted_ppp AS pt_iso_poss_weighted_ppp, pspt.iso_ppp AS pt_iso_ppp, pspt.iso_pts AS pt_iso_pts,
    pspt.iso_score_poss_pct AS pt_iso_score_poss_pct, pspt.iso_sf_poss_pct AS pt_iso_sf_poss_pct, pspt.iso_tov_poss_pct AS pt_iso_tov_poss_pct, pspt.iso_rppp AS pt_iso_rppp,
    pspt.transition_adj_percentile AS pt_transition_adj_percentile, pspt.transition_adj_poss AS pt_transition_adj_poss, pspt.transition_adj_ppp AS pt_transition_adj_ppp, pspt.transition_adj_pts AS pt_transition_adj_pts,
    pspt.transition_efg_pct AS pt_transition_efg_pct, pspt.transition_fga AS pt_transition_fga, pspt.transition_fga_per_g AS pt_transition_fga_per_g, pspt.transition_fga_percentile AS pt_transition_fga_percentile,
    pspt.transition_fgm AS pt_transition_fgm, pspt.transition_fgm_x AS pt_transition_fgm_x, pspt.transition_fg_pct AS pt_transition_fg_pct, pspt.transition_ft_poss_pct AS pt_transition_ft_poss_pct,
    pspt.transition_gp AS pt_transition_gp, pspt.transition_percentile AS pt_transition_percentile, pspt.transition_plusone_poss_pct AS pt_transition_plusone_poss_pct, pspt.transition_poe AS pt_transition_poe,
    pspt.transition_poe_per_g AS pt_transition_poe_per_g, pspt.transition_poe_rank AS pt_transition_poe_rank, pspt.transition_poss AS pt_transition_poss, pspt.transition_poss_per_g AS pt_transition_poss_per_g,
    pspt.transition_poss_percentile AS pt_transition_poss_percentile, pspt.transition_poss_weighted_ppp AS pt_transition_poss_weighted_ppp, pspt.transition_ppp AS pt_transition_ppp, pspt.transition_pts AS pt_transition_pts,
    pspt.transition_score_poss_pct AS pt_transition_score_poss_pct, pspt.transition_sf_poss_pct AS pt_transition_sf_poss_pct, pspt.transition_tov_poss_pct AS pt_transition_tov_poss_pct, pspt.transition_rppp AS pt_transition_rppp,

    pspt.mis_adj_percentile AS pt_mis_adj_percentile, pspt.mis_adj_poss AS pt_mis_adj_poss, pspt.mis_adj_ppp AS pt_mis_adj_ppp, pspt.mis_adj_pts AS pt_mis_adj_pts,
    pspt.mis_efg_pct AS pt_mis_efg_pct, pspt.mis_fga AS pt_mis_fga, pspt.mis_fga_per_g AS pt_mis_fga_per_g, pspt.mis_fga_percentile AS pt_mis_fga_percentile,
    pspt.mis_fgm AS pt_mis_fgm, pspt.mis_fgm_x AS pt_mis_fgm_x, pspt.mis_fg_pct AS pt_mis_fg_pct, pspt.mis_ft_poss_pct AS pt_mis_ft_poss_pct,
    pspt.mis_gp AS pt_mis_gp, pspt.mis_percentile AS pt_mis_percentile, pspt.mis_plusone_poss_pct AS pt_mis_plusone_poss_pct, pspt.mis_poe AS pt_mis_poe,
    pspt.mis_poe_per_g AS pt_mis_poe_per_g, pspt.mis_poe_rank AS pt_mis_poe_rank, pspt.mis_poss AS pt_mis_poss, pspt.mis_poss_per_g AS pt_mis_poss_per_g,
    pspt.mis_poss_percentile AS pt_mis_poss_percentile, pspt.mis_poss_weighted_ppp AS pt_mis_poss_weighted_ppp, pspt.mis_ppp AS pt_mis_ppp, pspt.mis_pts AS pt_mis_pts,
    pspt.mis_score_poss_pct AS pt_mis_score_poss_pct, pspt.mis_sf_poss_pct AS pt_mis_sf_poss_pct, pspt.mis_tov_poss_pct AS pt_mis_tov_poss_pct, pspt.mis_rppp AS pt_mis_rppp,

    pspt.postup_adj_percentile AS pt_postup_adj_percentile, pspt.postup_adj_poss AS pt_postup_adj_poss, pspt.postup_adj_ppp AS pt_postup_adj_ppp, pspt.postup_adj_pts AS pt_postup_adj_pts,
    pspt.postup_efg_pct AS pt_postup_efg_pct, pspt.postup_fga AS pt_postup_fga, pspt.postup_fga_per_g AS pt_postup_fga_per_g, pspt.postup_fga_percentile AS pt_postup_fga_percentile,
    pspt.postup_fgm AS pt_postup_fgm, pspt.postup_fgm_x AS pt_postup_fgm_x, pspt.postup_fg_pct AS pt_postup_fg_pct, pspt.postup_ft_poss_pct AS pt_postup_ft_poss_pct,
    pspt.postup_gp AS pt_postup_gp, pspt.postup_percentile AS pt_postup_percentile, pspt.postup_plusone_poss_pct AS pt_postup_plusone_poss_pct, pspt.postup_poe AS pt_postup_poe,
    pspt.postup_poe_per_g AS pt_postup_poe_per_g, pspt.postup_poe_rank AS pt_postup_poe_rank, pspt.postup_poss AS pt_postup_poss, pspt.postup_poss_per_g AS pt_postup_poss_per_g,
    pspt.postup_poss_percentile AS pt_postup_poss_percentile, pspt.postup_poss_weighted_ppp AS pt_postup_poss_weighted_ppp, pspt.postup_ppp AS pt_postup_ppp, pspt.postup_pts AS pt_postup_pts,
    pspt.postup_score_poss_pct AS pt_postup_score_poss_pct, pspt.postup_sf_poss_pct AS pt_postup_sf_poss_pct, pspt.postup_tov_poss_pct AS pt_postup_tov_poss_pct, pspt.postup_rppp AS pt_postup_rppp,

    pspt.spotup_adj_percentile AS pt_spotup_adj_percentile, pspt.spotup_adj_poss AS pt_spotup_adj_poss, pspt.spotup_adj_ppp AS pt_spotup_adj_ppp, pspt.spotup_adj_pts AS pt_spotup_adj_pts,
    pspt.spotup_efg_pct AS pt_spotup_efg_pct, pspt.spotup_fga AS pt_spotup_fga, pspt.spotup_fga_per_g AS pt_spotup_fga_per_g, pspt.spotup_fga_percentile AS pt_spotup_fga_percentile,
    pspt.spotup_fgm AS pt_spotup_fgm, pspt.spotup_fgm_x AS pt_spotup_fgm_x, pspt.spotup_fg_pct AS pt_spotup_fg_pct, pspt.spotup_ft_poss_pct AS pt_spotup_ft_poss_pct,
    pspt.spotup_gp AS pt_spotup_gp, pspt.spotup_percentile AS pt_spotup_percentile, pspt.spotup_plusone_poss_pct AS pt_spotup_plusone_poss_pct, pspt.spotup_poe AS pt_spotup_poe,
    pspt.spotup_poe_per_g AS pt_spotup_poe_per_g, pspt.spotup_poe_rank AS pt_spotup_poe_rank, pspt.spotup_poss AS pt_spotup_poss, pspt.spotup_poss_per_g AS pt_spotup_poss_per_g,
    pspt.spotup_poss_percentile AS pt_spotup_poss_percentile, pspt.spotup_poss_weighted_ppp AS pt_spotup_poss_weighted_ppp, pspt.spotup_ppp AS pt_spotup_ppp, pspt.spotup_pts AS pt_spotup_pts,
    pspt.spotup_score_poss_pct AS pt_spotup_score_poss_pct, pspt.spotup_sf_poss_pct AS pt_spotup_sf_poss_pct, pspt.spotup_tov_poss_pct AS pt_spotup_tov_poss_pct, pspt.spotup_rppp AS pt_spotup_rppp,
    pspt.handoff_adj_percentile AS pt_handoff_adj_percentile, pspt.handoff_adj_poss AS pt_handoff_adj_poss, pspt.handoff_adj_ppp AS pt_handoff_adj_ppp, pspt.handoff_adj_pts AS pt_handoff_adj_pts,
    pspt.handoff_efg_pct AS pt_handoff_efg_pct, pspt.handoff_fga AS pt_handoff_fga, pspt.handoff_fga_per_g AS pt_handoff_fga_per_g, pspt.handoff_fga_percentile AS pt_handoff_fga_percentile,
    pspt.handoff_fgm AS pt_handoff_fgm, pspt.handoff_fgm_x AS pt_handoff_fgm_x, pspt.handoff_fg_pct AS pt_handoff_fg_pct, pspt.handoff_ft_poss_pct AS pt_handoff_ft_poss_pct,
    pspt.handoff_gp AS pt_handoff_gp, pspt.handoff_percentile AS pt_handoff_percentile, pspt.handoff_plusone_poss_pct AS pt_handoff_plusone_poss_pct, pspt.handoff_poe AS pt_handoff_poe,
    pspt.handoff_poe_per_g AS pt_handoff_poe_per_g, pspt.handoff_poe_rank AS pt_handoff_poe_rank, pspt.handoff_poss AS pt_handoff_poss, pspt.handoff_poss_per_g AS pt_handoff_poss_per_g,
    pspt.handoff_poss_percentile AS pt_handoff_poss_percentile, pspt.handoff_poss_weighted_ppp AS pt_handoff_poss_weighted_ppp, pspt.handoff_ppp AS pt_handoff_ppp, pspt.handoff_pts AS pt_handoff_pts,
    pspt.handoff_score_poss_pct AS pt_handoff_score_poss_pct, pspt.handoff_sf_poss_pct AS pt_handoff_sf_poss_pct, pspt.handoff_tov_poss_pct AS pt_handoff_tov_poss_pct, pspt.handoff_rppp AS pt_handoff_rppp,

    pspt.offscreen_adj_percentile AS pt_offscreen_adj_percentile, pspt.offscreen_adj_poss AS pt_offscreen_adj_poss, pspt.offscreen_adj_ppp AS pt_offscreen_adj_ppp, pspt.offscreen_adj_pts AS pt_offscreen_adj_pts,
    pspt.offscreen_efg_pct AS pt_offscreen_efg_pct, pspt.offscreen_fga AS pt_offscreen_fga, pspt.offscreen_fga_per_g AS pt_offscreen_fga_per_g, pspt.offscreen_fga_percentile AS pt_offscreen_fga_percentile,
    pspt.offscreen_fgm AS pt_offscreen_fgm, pspt.offscreen_fgm_x AS pt_offscreen_fgm_x, pspt.offscreen_fg_pct AS pt_offscreen_fg_pct, pspt.offscreen_ft_poss_pct AS pt_offscreen_ft_poss_pct,
    pspt.offscreen_gp AS pt_offscreen_gp, pspt.offscreen_percentile AS pt_offscreen_percentile, pspt.offscreen_plusone_poss_pct AS pt_offscreen_plusone_poss_pct, pspt.offscreen_poe AS pt_offscreen_poe,
    pspt.offscreen_poe_per_g AS pt_offscreen_poe_per_g, pspt.offscreen_poe_rank AS pt_offscreen_poe_rank, pspt.offscreen_poss AS pt_offscreen_poss, pspt.offscreen_poss_per_g AS pt_offscreen_poss_per_g,
    pspt.offscreen_poss_percentile AS pt_offscreen_poss_percentile, pspt.offscreen_poss_weighted_ppp AS pt_offscreen_poss_weighted_ppp, pspt.offscreen_ppp AS pt_offscreen_ppp, pspt.offscreen_pts AS pt_offscreen_pts,
    pspt.offscreen_score_poss_pct AS pt_offscreen_score_poss_pct, pspt.offscreen_sf_poss_pct AS pt_offscreen_sf_poss_pct, pspt.offscreen_tov_poss_pct AS pt_offscreen_tov_poss_pct, pspt.offscreen_rppp AS pt_offscreen_rppp,

    pspt.cut_adj_percentile AS pt_cut_adj_percentile, pspt.cut_adj_poss AS pt_cut_adj_poss, pspt.cut_adj_ppp AS pt_cut_adj_ppp, pspt.cut_adj_pts AS pt_cut_adj_pts,
    pspt.cut_efg_pct AS pt_cut_efg_pct, pspt.cut_fga AS pt_cut_fga, pspt.cut_fga_per_g AS pt_cut_fga_per_g, pspt.cut_fga_percentile AS pt_cut_fga_percentile,
    pspt.cut_fgm AS pt_cut_fgm, pspt.cut_fgm_x AS pt_cut_fgm_x, pspt.cut_fg_pct AS pt_cut_fg_pct, pspt.cut_ft_poss_pct AS pt_cut_ft_poss_pct,
    pspt.cut_gp AS pt_cut_gp, pspt.cut_percentile AS pt_cut_percentile, pspt.cut_plusone_poss_pct AS pt_cut_plusone_poss_pct, pspt.cut_poe AS pt_cut_poe,
    pspt.cut_poe_per_g AS pt_cut_poe_per_g, pspt.cut_poe_rank AS pt_cut_poe_rank, pspt.cut_poss AS pt_cut_poss, pspt.cut_poss_per_g AS pt_cut_poss_per_g,
    pspt.cut_poss_percentile AS pt_cut_poss_percentile, pspt.cut_poss_weighted_ppp AS pt_cut_poss_weighted_ppp, pspt.cut_ppp AS pt_cut_ppp, pspt.cut_pts AS pt_cut_pts,
    pspt.cut_score_poss_pct AS pt_cut_score_poss_pct, pspt.cut_sf_poss_pct AS pt_cut_sf_poss_pct, pspt.cut_tov_poss_pct AS pt_cut_tov_poss_pct, pspt.cut_rppp AS pt_cut_rppp,

    pspt.putback_adj_percentile AS pt_putback_adj_percentile, pspt.putback_adj_poss AS pt_putback_adj_poss, pspt.putback_adj_ppp AS pt_putback_adj_ppp, pspt.putback_adj_pts AS pt_putback_adj_pts,
    pspt.putback_efg_pct AS pt_putback_efg_pct, pspt.putback_fga AS pt_putback_fga, pspt.putback_fga_per_g AS pt_putback_fga_per_g, pspt.putback_fga_percentile AS pt_putback_fga_percentile,
    pspt.putback_fgm AS pt_putback_fgm, pspt.putback_fgm_x AS pt_putback_fgm_x, pspt.putback_fg_pct AS pt_putback_fg_pct, pspt.putback_ft_poss_pct AS pt_putback_ft_poss_pct,
    pspt.putback_gp AS pt_putback_gp, pspt.putback_percentile AS pt_putback_percentile, pspt.putback_plusone_poss_pct AS pt_putback_plusone_poss_pct, pspt.putback_poe AS pt_putback_poe,
    pspt.putback_poe_per_g AS pt_putback_poe_per_g, pspt.putback_poe_rank AS pt_putback_poe_rank, pspt.putback_poss AS pt_putback_poss, pspt.putback_poss_per_g AS pt_putback_poss_per_g,
    pspt.putback_poss_percentile AS pt_putback_poss_percentile, pspt.putback_poss_weighted_ppp AS pt_putback_poss_weighted_ppp, pspt.putback_ppp AS pt_putback_ppp, pspt.putback_pts AS pt_putback_pts,
    pspt.putback_score_poss_pct AS pt_putback_score_poss_pct, pspt.putback_sf_poss_pct AS pt_putback_sf_poss_pct, pspt.putback_tov_poss_pct AS pt_putback_tov_poss_pct, pspt.putback_rppp AS pt_putback_rppp,
    pspt.rollman_adj_percentile AS pt_rollman_adj_percentile, pspt.rollman_adj_poss AS pt_rollman_adj_poss, pspt.rollman_adj_ppp AS pt_rollman_adj_ppp, pspt.rollman_adj_pts AS pt_rollman_adj_pts,
    pspt.rollman_efg_pct AS pt_rollman_efg_pct, pspt.rollman_fga AS pt_rollman_fga, pspt.rollman_fga_per_g AS pt_rollman_fga_per_g, pspt.rollman_fga_percentile AS pt_rollman_fga_percentile,
    pspt.rollman_fgm AS pt_rollman_fgm, pspt.rollman_fgm_x AS pt_rollman_fgm_x, pspt.rollman_fg_pct AS pt_rollman_fg_pct, pspt.rollman_ft_poss_pct AS pt_rollman_ft_poss_pct,
    pspt.rollman_gp AS pt_rollman_gp, pspt.rollman_percentile AS pt_rollman_percentile, pspt.rollman_plusone_poss_pct AS pt_rollman_plusone_poss_pct, pspt.rollman_poe AS pt_rollman_poe,
    pspt.rollman_poe_per_g AS pt_rollman_poe_per_g, pspt.rollman_poe_rank AS pt_rollman_poe_rank, pspt.rollman_poss AS pt_rollman_poss, pspt.rollman_poss_per_g AS pt_rollman_poss_per_g,
    pspt.rollman_poss_percentile AS pt_rollman_poss_percentile, pspt.rollman_poss_weighted_ppp AS pt_rollman_poss_weighted_ppp, pspt.rollman_ppp AS pt_rollman_ppp, pspt.rollman_pts AS pt_rollman_pts,
    pspt.rollman_score_poss_pct AS pt_rollman_score_poss_pct, pspt.rollman_sf_poss_pct AS pt_rollman_sf_poss_pct, pspt.rollman_tov_poss_pct AS pt_rollman_tov_poss_pct, pspt.rollman_rppp AS pt_rollman_rppp

                
    
    FROM pgf_with_avgs pgf
    JOIN players ps ON pgf.player_id = ps.player_id
    JOIN teams_game_features tgf ON pgf.game_id = tgf.game_id AND pgf.team_id = tgf.team_id
    JOIN teams_game_features otgf ON tgf.game_id = otgf.game_id AND tgf.opponent_team_id = otgf.team_id
    LEFT JOIN player_tracking_rs_for_join  rts  ON rts.player_id = pgf.player_id AND rts.season = pgf.season
    LEFT JOIN player_game_advanced_tracking pgat ON pgat.player_id = pgf.player_id AND pgat.game_id = pgf.game_id
    LEFT JOIN team_defense_tracking_rs_for_join tdt  ON tdt.team_id = pgf.opponent_team_id AND tdt.season   = pgf.season
    LEFT JOIN vegas_odds_for_join vo ON vo.game_id = pgf.game_id AND vo.team_id = pgf.team_id
    LEFT JOIN opponent_vs_player ovp ON pgf.game_id = ovp.game_id AND pgf.opponent_team_id = ovp.opponent_team_id AND ps.position_number = ovp.position_number
    LEFT JOIN player_on_court_rs_for_join   pon ON  pon.player_id = pgf.player_id AND pon.team_id = pgf.team_id AND pon.season = pgf.season
    LEFT JOIN player_off_court_rs_for_join  poff ON  poff.player_id = pgf.player_id AND poff.team_id = pgf.team_id AND poff.season = pgf.season
    LEFT JOIN opponent_position_onoff_rs  opo ON  opo.game_id         = pgf.game_id AND  opo.team_id         = pgf.team_id AND  opo.position_number = ps.position_number    
    LEFT JOIN star_players sp ON sp.season    = pgf.season AND sp.player_id = pgf.player_id
    LEFT JOIN star_out_flags sof ON sof.game_id = pgf.game_id AND sof.team_id = pgf.team_id
    LEFT JOIN player_season_playtypes_rs pspt ON pspt.player_id = pgf.player_id AND pspt.season    = pgf.season
"""