#!/usr/bin/env python3
"""
setup_vps.py — Auto-configures .env and data/accounts.json on the VPS.
Run once: python setup_vps.py
"""
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

ACCOUNTS = [
    {
        "id": "acc_V41xWypDvEiO",
        "email": "primary-account",
        "token": "w644QyNQBgHzyEUNTscIhcs8aOOQ7LwVXMOoCR+1irBLu8nEgZtdIM+eeom3PIzN",
        "cookies": "ext_name=ojplmecpdpgccookcobabopnaifgidhf; aws-waf-token=77ea2135-391b-4b4e-905d-3144fb712f2a:BQoAtM48QE4+AAAA:17C2UDZk+AeJG7JfzH+DCKLT1gbNgUjHsFnksLQ9z/74Kby1RYugrjlLZGkH5GepzDLs/xruhQ/AxrvnObDodsCp/SK3AMxKni9ZOxaDLQ0a/BRxZ0t33d7ExUkS16Xgkv5W8qeHNESAN7l9DQt+YcTlYIDKNk5LF8PFGXIUf3TaNEB3FZkznW90isdBYDk2xILHsznj2uIGndZAtokaZVS2ttC5/VnLLXnsyDojexma6cJeIZJdcrTYilDF9O5jXyooHfLpAV9wG39iVCpJR/9TKTPU0sUxtDaInuMLAdjlAGoqbRoBPpgOTrkh; smidV2=20260911140903c5c238312966bd2ab627fa9ca252e7bb00636f2ecf355e580; .thumbcache_6b2e5483f9d858d7c661c5e276b6a6ae=cm/J9fM8Nb3/P9CwTRUlVIMib28n3TTByLy7gLsdoi/N94cFs0IB27gdRxrdG1iVS5rcq/PCl0VbIDX2zoLCTg%3D%3D; ds_session_id=6067b66f0e344ae892f390fc2cefe3ad",
        "created_at": 1789117207,
        "updated_at": 1789117207,
    },
    {
        "id": "acc_ZlKAG0SS-OEh",
        "email": "acc2",
        "token": "nc0P7h2AyJ8bEhHynRtXQYroeJJ3/PVwsTokS2igujt+GIelxkrP0163/o6kRVbn",
        "cookies": "aws-waf-token=77ea2135-391b-4b4e-905d-3144fb712f2a:BQoAi45FFVwCAAAA:CQ4PZED4fon3MP+VaD4AbJ5mInd4QKSuelP1dnLxESONfW5p//BUOJZdWVEYDeJOlifYTyAO426AB9SXn6GCMvy4KseJ70FiEB3wuHkpJ1lLUnH+F71p9OPFHURiuDsum/CPOUbwjlAaEb3P/lr49zyMlaYqtLiKNTpttkaj+3+WNFwOC50O4qqnX3pF/wKEmciL+12lxw3BQg38QfXxRu3jsGPfkYMbmp+t4F3aZAvxMSP9W5ihrEn1ZQ2ZohZJ056HNDNVJ2DhPoYNiQxzY7e3UpnlcUMAl9fOu4n4wORwCziceuiDM0mBB3zQ; smidV2=20260911151948ecf7df301a1ccef3df0e7b3ef248533a00db04ca875fd5fa0; .thumbcache_6b2e5483f9d858d7c661c5e276b6a6ae=JAwS2sA2poOwhfHSARlDkOmTy9MPOV6eRCute7RUYUvyhoetJq53PpUj1xbqhWT5OXsuh0EB7X6IhVwpFvTMmA%3D%3D; ds_session_id=7e32a45075344e458bf299c9caa3f85e",
        "created_at": 1789120518,
        "updated_at": 1789120518,
    },
    {
        "id": "acc_XZw_f9quwYDR",
        "email": "acc3",
        "token": "MW5204H80Y2kjWKEN8EzGkH231x8AATQwY7oe16c6GXHAK/q1th4a4uFNLXDI8Wt",
        "cookies": "aws-waf-token=77ea2135-391b-4b4e-905d-3144fb712f2a:BQoAftlEvUcrAAAA:JXj/Q49eFCcRdP03rbaMk7belRjPeTQkGYB9micqmvYCR1fT2KvChN1QI2AtYAGI/xegaThTuzjWv5Tr+5sKXU166WCOjQ18r40+uUhjditBnWmg+k9nR5U/9br7rwaKfY99nQEJbJsulI/D+VFfRGXvrvmKsRtbMSZSvUjAFJJk8t/y4KTDU6THBUUY0hcx57orZZC0BCZxgWrIiih7XouwCcFPLpOATMWl+iBoycG5AdZbOgGjiDFUNpyB9lCaI5NzXh2x0owzU8GjMvDdP1wMuH/6QwwH1As4cnshBQaD9bt9WAJJsCmaNwu; smidV2=2026091115205734f42a586c5708900d0583b7b37f70fa0045d7a72a6a01bf0; .thumbcache_6b2e5483f9d858d7c661c5e276b6a6ae=D57qg+fryTxHx4WhiOjuJLLQxyKB+1kUPkxVBF+WlzO5jx7ipnuIrqpc2sghe1rTuJGf6UVVXUMNIrtu/5iy0g%3D%3D; ds_session_id=4f2d9f82995a43259b6f45320300a75c",
        "created_at": 1789120518,
        "updated_at": 1789120518,
    },
    {
        "id": "acc_XHqwlUsijJaK",
        "email": "acc4",
        "token": "POCMUiQhwgD7DxScq4AhxKeHmv4aivD4DchFfmDgpCP+MoWlZQbB+TsB6aFZKDZm",
        "cookies": "aws-waf-token=77ea2135-391b-4b4e-905d-3144fb712f2a:BQoAdw9FSTEJAAAA:U8cVOugpnESB83TLthid2yp74BmoVlJuQB9CYzOoXjzAbK2qKJsSEgWsa4A/cnrvQrWVMRBSRU/TnOiC2hlFP1qWhvjwF2S8ACwWHoDJeTgwEuaMXaoWgn/J88W70yOqA+5WHvdyXNQeNJn/MQADqXTlQXrKEsepuxCI8K8db97lSeKkmIbGmHMIB/A+pu6sihewyt8VQ07TkeBOnR76J/TWH8jug7qlHENnWf5gCJlOMKPH0SKt3opyMyYLwvlEYiEGkMRDQZVNTlTZOmWrIKqJP7Wjx55cEsDTWmxP0r1LYlfrXTN5TbZLy0mC; smidV2=20260911152331d97adad4575892f8421065ac31c9ea42007e84b72e4fe4ac0; .thumbcache_6b2e5483f9d858d7c661c5e276b6a6ae=QnDlxm7Wq8iXJN0qSA2N6S0HI3isUo1ReJfbn2Rvlduw7LAAIKtmAtWbo2qulwLs/lmSk8pCE1tq/y9N8q6txw%3D%3D; ds_session_id=69c441c99c214101b074dcdc239b2604",
        "created_at": 1789120519,
        "updated_at": 1789120519,
    },
]

ENV_CONTENT = f"""# Client API Auth
API_KEYS=sk-change-me
ALLOW_UNAUTHENTICATED_API=true

# Admin Password
DEEPSEEK_ADMIN_PASSWORD=change-me
ALLOW_INSECURE_PUBLIC_DEFAULTS=true

# Rate Limiting
ENABLE_RATE_LIMIT=false
CLIENT_RPM_PER_KEY=300
CLIENT_RPM_PER_IP=300

# Model Routing & Exposure
MODEL_ROUTES={{"gpt-6-astra":"default","gpt-6-astra-reasoner":"expert","cs":"default","cs-chat":"default","cs-reasoner":"expert"}}
MODEL_NAME=gpt-6-astra
MODEL_OWNER=openai

# Upstream OpenAI-compatible API
UPSTREAM_API_URL=http://43.153.6.116:8000/v1
UPSTREAM_API_KEY=
UPSTREAM_MODEL=deepseek-chat
UPSTREAM_MODE=api

# Network Bind & Port
HOST=0.0.0.0
PORT=8000

# Concurrency & Mode
DEEPSEEK_MAX_CONCURRENCY=4
MODE=auto
THINKING=auto
SEARCH=auto

# Account 1
DEEPSEEK_TOKEN_1={ACCOUNTS[0]["token"]}
DEEPSEEK_COOKIES_1={ACCOUNTS[0]["cookies"]}

# Account 2
DEEPSEEK_TOKEN_2={ACCOUNTS[1]["token"]}
DEEPSEEK_COOKIES_2={ACCOUNTS[1]["cookies"]}

# Account 3
DEEPSEEK_TOKEN_3={ACCOUNTS[2]["token"]}
DEEPSEEK_COOKIES_3={ACCOUNTS[2]["cookies"]}

# Account 4
DEEPSEEK_TOKEN_4={ACCOUNTS[3]["token"]}
DEEPSEEK_COOKIES_4={ACCOUNTS[3]["cookies"]}
"""

def main():
    # 1. Write data/accounts.json
    data_dir = BASE_DIR / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    accounts_file = data_dir / "accounts.json"
    accounts_file.write_text(json.dumps({"version": 1, "accounts": ACCOUNTS}, indent=2), encoding="utf-8")
    print(f"[+] Wrote {accounts_file} with {len(ACCOUNTS)} accounts.")

    # 2. Write .env
    env_file = BASE_DIR / ".env"
    env_file.write_text(ENV_CONTENT, encoding="utf-8")
    print(f"[+] Wrote {env_file} (PORT=8000, HOST=0.0.0.0, 4 accounts).")

    print("[OK] VPS setup complete!")

if __name__ == "__main__":
    main()
