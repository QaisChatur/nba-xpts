-- Supabase/Postgres schema for xPTS prototype (pre-frontend)

create table if not exists games (
  game_id text primary key,
  game_date date,
  season text
);

create table if not exists shots_xpts (
  game_id text not null,
  game_event_id bigint not null,
  game_date date,
  player_id bigint,
  player_name text,
  team_id bigint,
  team_name text,
  season text,
  period int,
  minutes_remaining int,
  seconds_remaining int,
  seconds_elapsed_in_game numeric,
  shot_type text,
  action_type text,
  shot_zone_basic text,
  shot_zone_area text,
  shot_zone_range text,
  shot_distance numeric,
  loc_x numeric,
  loc_y numeric,
  shot_made_flag int,
  shot_value int,
  actual_points numeric,
  xfg_pct_player_specific numeric,
  xpts_player_specific numeric,
  points_above_expected numeric,
  distance_bin text,
  is_deep_3 int,
  is_corner_3 int,
  is_restricted_area int,
  is_midrange int,
  is_paint_non_ra int,
  player_zone_action_fg_smoothed numeric,
  player_distance_bin_fg_smoothed numeric,
  player_action_fg_smoothed numeric,
  player_overall_fg_smoothed numeric,
  clip_event_url text,
  primary key (game_id, game_event_id, player_id, team_id)
);

create index if not exists idx_shots_xpts_game on shots_xpts (game_id);
create index if not exists idx_shots_xpts_season on shots_xpts (season);
create index if not exists idx_shots_xpts_player on shots_xpts (player_id);
create index if not exists idx_shots_xpts_zone on shots_xpts (shot_zone_basic);
create index if not exists idx_shots_xpts_shot_type on shots_xpts (shot_type);

create table if not exists game_flow_events (
  game_id text not null,
  game_event_id bigint not null,
  game_date date,
  player_id bigint,
  player_name text,
  team_id bigint,
  team_name text,
  season text,
  period int,
  minutes_remaining int,
  seconds_remaining int,
  seconds_elapsed_in_game numeric,
  shot_type text,
  action_type text,
  shot_zone_basic text,
  shot_zone_area text,
  shot_zone_range text,
  shot_distance numeric,
  loc_x numeric,
  loc_y numeric,
  shot_made_flag int,
  shot_value int,
  actual_points numeric,
  xfg_pct_player_specific numeric,
  xpts_player_specific numeric,
  points_above_expected numeric,
  distance_bin text,
  is_deep_3 int,
  is_corner_3 int,
  is_restricted_area int,
  is_midrange int,
  is_paint_non_ra int,
  player_zone_action_fg_smoothed numeric,
  player_distance_bin_fg_smoothed numeric,
  player_action_fg_smoothed numeric,
  player_overall_fg_smoothed numeric,
  clip_event_url text,
  actual_cum_points_team numeric,
  xpts_cum_team numeric,
  primary key (game_id, game_event_id, team_id)
);

create index if not exists idx_game_flow_game on game_flow_events (game_id);
create index if not exists idx_game_flow_game_period on game_flow_events (game_id, period, seconds_elapsed_in_game);

create table if not exists player_zone_shottype_summary (
  season text not null,
  player_id bigint not null,
  player_name text,
  team_name text,
  shot_type text not null,
  shot_zone_basic text not null,
  shots bigint,
  makes bigint,
  actual_points numeric,
  expected_points numeric,
  avg_xfg numeric,
  fg_pct numeric,
  actual_points_per_shot numeric,
  xpts_per_shot numeric,
  points_above_expected numeric,
  points_above_expected_per_shot numeric,
  primary key (season, player_id, shot_type, shot_zone_basic)
);

create index if not exists idx_player_zone_summary_player on player_zone_shottype_summary (player_id);
create index if not exists idx_player_zone_summary_zone on player_zone_shottype_summary (shot_zone_basic);

