create extension if not exists pgcrypto;

create table if not exists profiles (
  owner_id text primary key,
  profile_json text not null default '{}',
  created_at timestamptz not null default timezone('utc', now()),
  updated_at timestamptz not null default timezone('utc', now())
);
alter table profiles enable row level security;

create table if not exists conversations (
  id text primary key,
  owner_id text not null,
  title text not null,
  system_preset text not null,
  system_prompt text not null default '',
  settings_json text not null default '{}',
  message_count integer not null default 0,
  preview text not null default '',
  created_at timestamptz not null default timezone('utc', now()),
  updated_at timestamptz not null default timezone('utc', now())
);
alter table conversations enable row level security;

create index if not exists idx_conversations_owner_updated on conversations(owner_id, updated_at desc);

create table if not exists messages (
  id text primary key,
  conversation_id text not null,
  owner_id text not null,
  role text not null,
  content text not null,
  metadata_json text not null default '{}',
  created_at timestamptz not null default timezone('utc', now())
);
alter table messages enable row level security;

create index if not exists idx_messages_conversation_created on messages(conversation_id, created_at asc);

create table if not exists memory_items (
  id text primary key,
  owner_id text not null,
  memory_key text not null,
  category text not null,
  summary text not null,
  context_json text not null default '{}',
  created_at timestamptz not null default timezone('utc', now()),
  updated_at timestamptz not null default timezone('utc', now()),
  last_used_at timestamptz not null default timezone('utc', now())
);
alter table memory_items enable row level security;

create unique index if not exists idx_memory_items_owner_key on memory_items(owner_id, memory_key);
create index if not exists idx_memory_items_owner_last_used on memory_items(owner_id, last_used_at desc);

create table if not exists conversation_summaries (
  conversation_id text primary key,
  owner_id text not null,
  summary_json text not null default '{}',
  created_at timestamptz not null default timezone('utc', now()),
  updated_at timestamptz not null default timezone('utc', now())
);
alter table conversation_summaries enable row level security;

create table if not exists api_keys (
  id text primary key,
  owner_id text not null,
  label text not null,
  key_prefix text not null,
  key_hash text not null unique,
  rate_limit_json text not null default '{}',
  created_at timestamptz not null default timezone('utc', now()),
  last_used_at timestamptz,
  revoked_at timestamptz,
  usage_count integer not null default 0
);
alter table api_keys enable row level security;

create index if not exists idx_api_keys_owner_created on api_keys(owner_id, created_at desc);

create table if not exists request_logs (
  id text primary key,
  owner_id text not null default '',
  conversation_id text,
  api_key_id text,
  route text not null,
  status_code integer not null,
  latency_ms double precision not null,
  metadata_json text not null default '{}',
  created_at timestamptz not null default timezone('utc', now())
);
alter table request_logs enable row level security;

create index if not exists idx_request_logs_api_key_created on request_logs(api_key_id, created_at desc);
