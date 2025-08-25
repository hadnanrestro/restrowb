# test_openai.py
from openai import OpenAI

# 🔑 Put your key here directly
API_KEY = "sk-"

def main():
    try:
        client = OpenAI(api_key=API_KEY)

        # Make a minimal test request
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=5,
            temperature=0,
        )

        msg = (resp.choices[0].message.content or "").strip()
        print("✅ Key works! Sample response:", msg)

    except Exception as e:
        print("❌ Key test failed:", str(e))

if __name__ == "__main__":
    main()