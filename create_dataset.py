"""
create_dataset.py — Generate a South African 11-language dataset and push to Hugging Face.

Usage:
    python create_dataset.py --skip-upload               # local JSONL only
    python create_dataset.py --username <hf_user> \
                             --repo-name mzansilm-sa11 \
                             --token hf_...              # translate + push
"""

import argparse
import copy
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

# ---------------------------------------------------------------------------
# Language table — {name: google_translate_code}
# nso / ss / ve / nr have limited or no Google Translate support;
# the script detects this at runtime and skips gracefully.
# ---------------------------------------------------------------------------
LANGUAGES: dict[str, str] = {
    "english":          "en",
    "afrikaans":        "af",
    "zulu":             "zu",
    "xhosa":            "xh",
    "southern_sotho":   "st",
    "tswana":           "tn",
    "northern_sotho":   "nso",
    "tsonga":           "ts",
    "swati":            "ss",
    "venda":            "ve",
    "southern_ndebele": "nr",
}

TOOL_SYSTEM_PROMPT = (
    "You are a helpful assistant with access to the following tools:\n"
    "- get_weather: Get current weather for a location. Args: location (string, required)\n"
    "- search_web: Search the web for information. Args: query (string, required)\n"
    "- calculate: Evaluate a mathematical expression. Args: expression (string, required)\n"
    "\n"
    "To call a tool, output exactly:\n"
    '<tool_call>{"name": "tool_name", "arguments": {"arg": "value"}}</tool_call>\n'
    "\n"
    "After receiving a tool result inside a <tool_result> block, continue the conversation naturally."
)

PLAIN_SYSTEM_PROMPT = (
    "You are MzansiLM, a helpful AI assistant with deep knowledge of "
    "South African culture, history, languages, and daily life."
)

# Regex that matches <tool_call>...</tool_call> and <tool_result>...</tool_result>
# These blocks are preserved verbatim during translation.
_TAG_PATTERN = re.compile(
    r"(<tool_call>.*?</tool_call>|<tool_result>.*?</tool_result>)",
    re.DOTALL,
)

# Languages confirmed unsupported at runtime — skip without retrying
CONFIRMED_UNSUPPORTED: set[str] = set()


# ---------------------------------------------------------------------------
# English source examples
# ---------------------------------------------------------------------------

def get_english_examples() -> list[dict]:
    S = PLAIN_SYSTEM_PROMPT
    T = TOOL_SYSTEM_PROMPT
    return [
        # --- Greetings and Ubuntu philosophy ---
        {
            "messages": [
                {"role": "system", "content": S},
                {"role": "user", "content": "What does 'Ubuntu' mean?"},
                {"role": "assistant", "content": (
                    "Ubuntu is a Nguni Bantu concept meaning 'I am because we are.' "
                    "It captures the idea that a person's humanity is bound up with that of others — "
                    "that we exist and thrive through our relationships and community. "
                    "It is central to the cultural philosophy of many South African peoples, "
                    "especially Zulu and Xhosa, and was a guiding principle in post-apartheid reconciliation."
                )},
            ]
        },
        {
            "messages": [
                {"role": "system", "content": S},
                {"role": "user", "content": "How do I greet someone in Zulu?"},
                {"role": "assistant", "content": (
                    "To greet one person in Zulu, say **Sawubona** (sah-woo-BOH-nah), which literally means "
                    "'I see you' — acknowledging the other person's humanity. The reply is **Yebo** (yes) or "
                    "**Ngikhona** (I am here). To greet a group, use **Sanibonani**. "
                    "Zulu greetings carry deep respect and are an important social ritual."
                )},
            ]
        },
        {
            "messages": [
                {"role": "system", "content": S},
                {"role": "user", "content": "How is Ubuntu practised in modern South Africa?"},
                {"role": "assistant", "content": (
                    "Ubuntu shaped post-apartheid South Africa profoundly. Nelson Mandela and Archbishop "
                    "Desmond Tutu drew on it to build the Truth and Reconciliation Commission — choosing "
                    "restorative justice over retribution. In daily life it shows up as communal care, "
                    "sharing resources with neighbours, and the idea that individual success is incomplete "
                    "without community wellbeing. Many SA companies and government bodies formally incorporate "
                    "Ubuntu into their values."
                )},
            ]
        },
        {
            "messages": [
                {"role": "system", "content": S},
                {"role": "user", "content": "How do I say hello in Afrikaans?"},
                {"role": "assistant", "content": (
                    "The common Afrikaans greetings are **Hallo** (hello) and **Goeie dag** (good day). "
                    "For good morning say **Goeie môre**, for good evening **Goeie naand**. "
                    "Afrikaans hospitality is known for warmth — guests are typically offered tea or coffee "
                    "immediately, and refusing can be considered rude. The phrase **Lekker dag!** "
                    "(have a nice day) is widely used across all communities in South Africa."
                )},
            ]
        },

        # --- SA Geography and Landmarks ---
        {
            "messages": [
                {"role": "system", "content": S},
                {"role": "user", "content": "What are the top things to see in Cape Town?"},
                {"role": "assistant", "content": (
                    "Cape Town's top attractions include:\n\n"
                    "1. **Table Mountain** — iconic flat-topped mountain, cable car or hike to the top.\n"
                    "2. **Robben Island** — where Nelson Mandela was imprisoned; now a UNESCO World Heritage Site.\n"
                    "3. **V&A Waterfront** — vibrant harbour with restaurants, shops, and the Two Oceans Aquarium.\n"
                    "4. **Cape Point and Cape of Good Hope** — dramatic cliffs where Atlantic and Indian Oceans meet.\n"
                    "5. **Boulders Beach** — home to a colony of African penguins near Simon's Town.\n\n"
                    "The Cape Winelands (Stellenbosch, Franschhoek) are also a short drive away."
                )},
            ]
        },
        {
            "messages": [
                {"role": "system", "content": S},
                {"role": "user", "content": "What is the difference between living in Johannesburg and Cape Town?"},
                {"role": "assistant", "content": (
                    "**Johannesburg (Joburg/Egoli)** is South Africa's economic engine. It sits on the Highveld "
                    "at 1,750 m altitude, giving it hot summers with afternoon thunderstorms and mild winters. "
                    "The pace is fast, business-focused, and multicultural. Traffic is notoriously heavy.\n\n"
                    "**Cape Town** is coastal, scenic, and has a Mediterranean climate — hot dry summers, "
                    "wet winters. It has a strong creative and tourism sector, but also a high cost of living "
                    "and seasonal employment. The city is known for its natural beauty and outdoor lifestyle.\n\n"
                    "Many South Africans say Joburg is where you go to work and Cape Town is where you go to live."
                )},
            ]
        },
        {
            "messages": [
                {"role": "system", "content": S},
                {"role": "user", "content": "Tell me about Kruger National Park."},
                {"role": "assistant", "content": (
                    "Kruger National Park is one of Africa's largest game reserves, covering roughly 2 million "
                    "hectares in Limpopo and Mpumalanga. It is home to the Big Five (lion, leopard, elephant, "
                    "rhino, buffalo) and over 500 bird species.\n\n"
                    "**Best time to visit:** May–September (dry winter). Vegetation thins out and animals "
                    "congregate around water sources, making sightings much easier.\n\n"
                    "You can self-drive on well-maintained tar and gravel roads, or join guided ranger drives "
                    "from the rest camps. Accommodation ranges from basic campsites to luxury private lodges "
                    "in the Greater Kruger concessions."
                )},
            ]
        },
        {
            "messages": [
                {"role": "system", "content": S},
                {"role": "user", "content": "What is Durban's Golden Mile?"},
                {"role": "assistant", "content": (
                    "The Golden Mile is Durban's famous beachfront strip along the Indian Ocean in KwaZulu-Natal. "
                    "It stretches several kilometres from uShaka Marine World in the south to the Bay of Plenty "
                    "beach in the north. The area features a promenade, surf-friendly waves, water slides, and "
                    "is popular with both locals and tourists year-round thanks to Durban's subtropical climate. "
                    "Durban has the busiest port in sub-Saharan Africa and a large Indian-origin population "
                    "that has greatly influenced its food culture."
                )},
            ]
        },

        # --- Food ---
        {
            "messages": [
                {"role": "system", "content": S},
                {"role": "user", "content": "What is a braai and how is it different from a regular barbecue?"},
                {"role": "assistant", "content": (
                    "A **braai** (from Afrikaans: to grill/roast) is far more than cooking — it is a South "
                    "African social institution. Key differences from a standard BBQ:\n\n"
                    "- **Fire:** Real wood or hardwood charcoal is essential. Gas braais are frowned upon by purists.\n"
                    "- **Meat:** Boerewors (spiced farmer's sausage), lamb chops, chicken, sosaties (kebabs), and "
                    "   steak are classics.\n"
                    "- **Social:** A braai is an all-afternoon event. You bring salads (potato salad, coleslaw), "
                    "   garlic bread, and cold drinks. Everyone gathers around the fire.\n"
                    "- **Cultural:** It transcends race and class — the braai is one of South Africa's great "
                    "   unifying rituals. National Braai Day falls on Heritage Day, 24 September."
                )},
            ]
        },
        {
            "messages": [
                {"role": "system", "content": S},
                {"role": "user", "content": "What is biltong and how is it made?"},
                {"role": "assistant", "content": (
                    "Biltong is a South African cured meat snack, usually made from beef or game (kudu, springbok). "
                    "Unlike jerky, it is **air-dried, not cooked** — the key distinction.\n\n"
                    "**Basic process:**\n"
                    "1. Cut meat into strips along the grain.\n"
                    "2. Marinate in vinegar (tenderises and adds tang).\n"
                    "3. Coat with salt, sugar, coriander, black pepper, and optional chilli.\n"
                    "4. Hang in a well-ventilated space (or biltong dryer box) for 3–7 days until desired dryness.\n\n"
                    "The result is a high-protein, low-carb snack. Wet biltong is softer; dry biltong is "
                    "harder and more concentrated. It is a staple in South African households and sports culture."
                )},
            ]
        },
        {
            "messages": [
                {"role": "system", "content": S},
                {"role": "user", "content": "What is bunny chow?"},
                {"role": "assistant", "content": (
                    "Bunny chow is a beloved street food originating from the Indian community in Durban, "
                    "KwaZulu-Natal. It consists of a **hollowed-out loaf of white bread filled with curry** — "
                    "typically mutton, chicken, or bean curry. The bread 'lid' is placed on top.\n\n"
                    "It originated in the 1940s when Indian workers needed a portable lunch that didn't require "
                    "cutlery. The bread soaks up the curry sauce, making it deliciously messy. "
                    "A quarter loaf (quarter bunny) is a standard serving. Despite the name, it contains no rabbit."
                )},
            ]
        },
        {
            "messages": [
                {"role": "system", "content": S},
                {"role": "user", "content": "How do I make bobotie?"},
                {"role": "assistant", "content": (
                    "Bobotie is South Africa's national dish — a Cape Malay spiced mince bake with an egg custard topping.\n\n"
                    "**Ingredients (serves 4):**\n"
                    "- 500 g beef or lamb mince\n"
                    "- 1 onion (chopped), 2 cloves garlic\n"
                    "- 2 tbsp curry powder, 1 tsp turmeric\n"
                    "- 2 tbsp apricot jam or chutney\n"
                    "- 2 slices white bread soaked in milk, then squeezed\n"
                    "- 2 eggs + 200 ml milk for the topping\n"
                    "- Bay leaves, salt and pepper\n\n"
                    "**Method:** Fry onion and garlic, add mince and spices, cook through. "
                    "Mix in the bread, jam, and seasoning. Press into a baking dish. "
                    "Whisk eggs with milk and pour over. Lay bay leaves on top. "
                    "Bake at 180°C for 35–40 minutes until the custard is set and golden. "
                    "Serve with yellow rice and chutney."
                )},
            ]
        },

        # --- Economy and Loadshedding ---
        {
            "messages": [
                {"role": "system", "content": S},
                {"role": "user", "content": "What is loadshedding?"},
                {"role": "assistant", "content": (
                    "Loadshedding is the controlled, rotational switching off of electricity to parts of the "
                    "country when South Africa's power grid cannot meet demand. It is managed by Eskom, the "
                    "state-owned power utility.\n\n"
                    "**Stages:** Stage 1 (lightest) to Stage 8 (most severe). Each stage indicates how many "
                    "gigawatts must be shed. Stage 6 means roughly 6,000 MW removed from the grid, resulting "
                    "in up to 12+ hours of outages per day for many households.\n\n"
                    "**Causes:** Ageing coal power stations, poor maintenance, diesel shortages at open-cycle "
                    "gas turbines, and years of under-investment.\n\n"
                    "Loadshedding has severely impacted South Africa's economy and daily life — businesses use "
                    "generators or solar/battery backup, and schedules are tracked via apps like EskomSePush."
                )},
            ]
        },
        {
            "messages": [
                {"role": "system", "content": S},
                {"role": "user", "content": "What is SASSA and who qualifies for grants?"},
                {"role": "assistant", "content": (
                    "SASSA (South African Social Security Agency) administers social grants on behalf of the "
                    "Department of Social Development.\n\n"
                    "**Main grants:**\n"
                    "- **Child Support Grant** — R530/month per child (under 18) for low-income caregivers.\n"
                    "- **Old Age Pension** — R2,190/month for citizens 60+ who pass a means test.\n"
                    "- **Disability Grant** — R2,190/month for people with a temporary or permanent disability.\n"
                    "- **Foster Child Grant** — R1,130/month for children in foster care.\n"
                    "- **Social Relief of Distress (SRD)** — R370/month, introduced during COVID-19.\n\n"
                    "Applications are made at SASSA offices. You can also apply online at sassa.gov.za or via "
                    "the SASSA WhatsApp line."
                )},
            ]
        },

        # --- Tool calls: Weather ---
        {
            "messages": [
                {"role": "system", "content": T},
                {"role": "user", "content": "What's the weather like in Port Elizabeth today?"},
                {"role": "assistant", "content": '<tool_call>{"name": "get_weather", "arguments": {"location": "Gqeberha"}}</tool_call>'},
                {"role": "user", "content": '<tool_result>{"temperature": 19, "condition": "Overcast with coastal wind", "humidity": 72, "wind_kph": 25}</tool_result>'},
                {"role": "assistant", "content": (
                    "The weather in Gqeberha (Port Elizabeth) today is overcast with a coastal wind at 25 km/h. "
                    "The temperature is 19°C with 72% humidity — typical for the Windy City. "
                    "A light jacket would be a good idea if you're heading out."
                )},
            ]
        },
        {
            "messages": [
                {"role": "system", "content": T},
                {"role": "user", "content": "What's the weather in Bloemfontein right now?"},
                {"role": "assistant", "content": '<tool_call>{"name": "get_weather", "arguments": {"location": "Bloemfontein"}}</tool_call>'},
                {"role": "user", "content": '<tool_result>{"temperature": 14, "condition": "Clear and cold", "humidity": 30, "wind_kph": 10}</tool_result>'},
                {"role": "assistant", "content": (
                    "Bloemfontein is currently clear and cold at 14°C with low humidity (30%) and a light breeze. "
                    "The city of roses sits in the Free State Highveld, known for its crisp, dry winters. "
                    "Pack a coat if you're going out."
                )},
            ]
        },
        {
            "messages": [
                {"role": "system", "content": T},
                {"role": "user", "content": "I need to compare the weather in Johannesburg and Durban. Which city is warmer?"},
                {"role": "assistant", "content": (
                    "Let me check both cities for you.\n"
                    '<tool_call>{"name": "get_weather", "arguments": {"location": "Johannesburg"}}</tool_call>'
                )},
                {"role": "user", "content": '<tool_result>{"temperature": 18, "condition": "Sunny", "humidity": 40}</tool_result>'},
                {"role": "assistant", "content": '<tool_call>{"name": "get_weather", "arguments": {"location": "Durban"}}</tool_call>'},
                {"role": "user", "content": '<tool_result>{"temperature": 27, "condition": "Humid and warm", "humidity": 82}</tool_result>'},
                {"role": "assistant", "content": (
                    "Durban is significantly warmer at 27°C compared to Johannesburg's 18°C. Durban is also much "
                    "more humid (82% vs 40%), giving it that coastal tropical feel. Johannesburg is sunny but cooler "
                    "due to its higher elevation on the Highveld."
                )},
            ]
        },
        {
            "messages": [
                {"role": "system", "content": T},
                {"role": "user", "content": "Should I bring an umbrella to Pretoria tomorrow?"},
                {"role": "assistant", "content": '<tool_call>{"name": "get_weather", "arguments": {"location": "Pretoria"}}</tool_call>'},
                {"role": "user", "content": '<tool_result>{"temperature": 24, "condition": "Thunderstorms expected in the afternoon", "humidity": 75, "rain_chance_percent": 80}</tool_result>'},
                {"role": "assistant", "content": (
                    "Yes, definitely bring an umbrella. Pretoria is expecting afternoon thunderstorms tomorrow with "
                    "an 80% chance of rain. The morning should be fine at 24°C, but conditions will deteriorate in "
                    "the afternoon. The high humidity (75%) already hints at the unstable air typical of Highveld "
                    "summer storms."
                )},
            ]
        },

        # --- Tool calls: Currency and calculations ---
        {
            "messages": [
                {"role": "system", "content": T},
                {"role": "user", "content": "How much is 1 USD in ZAR and what would 500 USD be?"},
                {"role": "assistant", "content": '<tool_call>{"name": "search_web", "arguments": {"query": "current USD to ZAR exchange rate"}}</tool_call>'},
                {"role": "user", "content": '<tool_result>{"results": [{"title": "USD/ZAR Exchange Rate", "snippet": "1 USD = 18.65 ZAR as of today."}]}</tool_result>'},
                {"role": "assistant", "content": '<tool_call>{"name": "calculate", "arguments": {"expression": "500 * 18.65"}}</tool_call>'},
                {"role": "user", "content": '<tool_result>{"result": 9325.0}</tool_result>'},
                {"role": "assistant", "content": (
                    "The current exchange rate is 1 USD = R18.65. So 500 USD would be **R9,325**. "
                    "Keep in mind exchange rates fluctuate, so check a live source before any actual transactions."
                )},
            ]
        },
        {
            "messages": [
                {"role": "system", "content": T},
                {"role": "user", "content": "What's 15% tip on a R340 restaurant bill?"},
                {"role": "assistant", "content": '<tool_call>{"name": "calculate", "arguments": {"expression": "340 * 0.15"}}</tool_call>'},
                {"role": "user", "content": '<tool_result>{"result": 51.0}</tool_result>'},
                {"role": "assistant", "content": "A 15% tip on R340 comes to R51. So your total would be R391."},
            ]
        },
        {
            "messages": [
                {"role": "system", "content": T},
                {"role": "user", "content": "What is R1000 worth in British pounds?"},
                {"role": "assistant", "content": '<tool_call>{"name": "search_web", "arguments": {"query": "current ZAR to GBP exchange rate"}}</tool_call>'},
                {"role": "user", "content": '<tool_result>{"results": [{"title": "ZAR/GBP Exchange Rate", "snippet": "1 ZAR = 0.043 GBP as of today."}]}</tool_result>'},
                {"role": "assistant", "content": '<tool_call>{"name": "calculate", "arguments": {"expression": "1000 * 0.043"}}</tool_call>'},
                {"role": "user", "content": '<tool_result>{"result": 43.0}</tool_result>'},
                {"role": "assistant", "content": (
                    "At today's rate (1 ZAR = £0.043), R1,000 is worth approximately **£43**. "
                    "Exchange rates vary daily — always verify before transferring money."
                )},
            ]
        },
        {
            "messages": [
                {"role": "system", "content": T},
                {"role": "user", "content": "If I earn R25,000 a month, what is that annually and in USD?"},
                {"role": "assistant", "content": '<tool_call>{"name": "calculate", "arguments": {"expression": "25000 * 12"}}</tool_call>'},
                {"role": "user", "content": '<tool_result>{"result": 300000}</tool_result>'},
                {"role": "assistant", "content": '<tool_call>{"name": "search_web", "arguments": {"query": "current ZAR to USD exchange rate"}}</tool_call>'},
                {"role": "user", "content": '<tool_result>{"results": [{"title": "ZAR/USD Exchange Rate", "snippet": "1 ZAR = 0.054 USD as of today."}]}</tool_result>'},
                {"role": "assistant", "content": '<tool_call>{"name": "calculate", "arguments": {"expression": "300000 * 0.054"}}</tool_call>'},
                {"role": "user", "content": '<tool_result>{"result": 16200.0}</tool_result>'},
                {"role": "assistant", "content": (
                    "R25,000/month equals **R300,000 per year**. At today's exchange rate (1 ZAR ≈ $0.054), "
                    "that's approximately **$16,200 USD annually**."
                )},
            ]
        },

        # --- History ---
        {
            "messages": [
                {"role": "system", "content": S},
                {"role": "user", "content": "Tell me about Nelson Mandela's life."},
                {"role": "assistant", "content": (
                    "Nelson Rolihlahla Mandela (1918–2013) was born in Mvezo in the Transkei (Eastern Cape). "
                    "He studied law at Fort Hare and became a founding member of the ANC Youth League in 1944. "
                    "He co-founded South Africa's first Black law firm with Oliver Tambo in 1952.\n\n"
                    "In 1964, Mandela was convicted of sabotage and sentenced to life imprisonment during the "
                    "Rivonia Trial. He spent 27 years on Robben Island and other prisons before being released "
                    "on 11 February 1990 by President F.W. de Klerk.\n\n"
                    "In 1994 he became South Africa's first democratically elected president after the country's "
                    "first universal elections. He and de Klerk shared the Nobel Peace Prize in 1993. "
                    "Mandela served one term, stepping down in 1999. He passed away on 5 December 2013."
                )},
            ]
        },
        {
            "messages": [
                {"role": "system", "content": S},
                {"role": "user", "content": "What does Freedom Day (27 April) commemorate in South Africa?"},
                {"role": "assistant", "content": (
                    "Freedom Day on 27 April commemorates the first democratic elections held in South Africa "
                    "in 1994 — the day when citizens of all races voted for the first time under universal "
                    "suffrage. It was the formal end of apartheid rule.\n\n"
                    "The elections ran from 27–29 April 1994 and resulted in Nelson Mandela and the ANC winning "
                    "with 62.6% of the vote. Mandela was inaugurated as president on 10 May 1994.\n\n"
                    "Freedom Day is a national public holiday and is marked with events celebrating democracy, "
                    "human rights, and South Africa's constitution."
                )},
            ]
        },
        {
            "messages": [
                {"role": "system", "content": S},
                {"role": "user", "content": "What happened during the Soweto Uprising in 1976?"},
                {"role": "assistant", "content": (
                    "On 16 June 1976, thousands of Black school students in Soweto marched to protest the "
                    "apartheid government's decree that Afrikaans had to be used as the medium of instruction "
                    "in schools alongside English. Students saw this as an imposition of the oppressor's language.\n\n"
                    "Police opened fire on the peaceful march. Hector Pieterson, a 12-year-old student, became "
                    "the iconic face of the uprising after a photograph of him being carried while dying was "
                    "published worldwide, galvanising international opposition to apartheid.\n\n"
                    "The uprisings spread across the country, resulting in hundreds of deaths. It significantly "
                    "accelerated global pressure on the apartheid regime. 16 June is now commemorated as "
                    "**Youth Day**, a national public holiday."
                )},
            ]
        },
        {
            "messages": [
                {"role": "system", "content": S},
                {"role": "user", "content": "What was the Truth and Reconciliation Commission?"},
                {"role": "assistant", "content": (
                    "The Truth and Reconciliation Commission (TRC) was established in 1996 under the chairmanship "
                    "of Archbishop Desmond Tutu. Its mandate was to investigate human rights violations committed "
                    "during the apartheid era (1960–1994) from all sides.\n\n"
                    "**Key principles:**\n"
                    "- Perpetrators could apply for **amnesty** if they made full, honest disclosure of their crimes.\n"
                    "- Victims could testify publicly, giving them a voice and acknowledgement.\n"
                    "- The goal was restorative justice — healing the nation — rather than punitive prosecution.\n\n"
                    "The TRC held public hearings across the country from 1996 to 1998, documenting thousands of "
                    "cases. It was widely studied as a model for transitional justice, though critics argued it "
                    "did not go far enough in delivering justice to victims."
                )},
            ]
        },

        # --- Sports ---
        {
            "messages": [
                {"role": "system", "content": S},
                {"role": "user", "content": "Tell me about the Springboks rugby team."},
                {"role": "assistant", "content": (
                    "The Springboks are South Africa's national rugby union team, the most successful in Rugby "
                    "World Cup history. They wear green and gold jerseys.\n\n"
                    "**World Cup wins:** 1995 (South Africa as hosts), 2007 (France), 2019 (Japan), 2023 (France) "
                    "— making them the first team to win four Rugby World Cups.\n\n"
                    "The 2019 victory was particularly historic as Siya Kolisi became the first Black captain to "
                    "lift the Webb Ellis Cup. The win was seen as a symbol of a unified South Africa.\n\n"
                    "The Springboks compete in The Rugby Championship alongside New Zealand (All Blacks), "
                    "Australia (Wallabies), and Argentina (Pumas)."
                )},
            ]
        },
        {
            "messages": [
                {"role": "system", "content": S},
                {"role": "user", "content": "What is Bafana Bafana?"},
                {"role": "assistant", "content": (
                    "Bafana Bafana (Zulu: 'the boys, the boys') is South Africa's national men's football (soccer) "
                    "team. Their greatest achievement was winning the Africa Cup of Nations (AFCON) in 1996 as "
                    "hosts, defeating Tunisia 2–0 in the final.\n\n"
                    "South Africa also hosted the **2010 FIFA World Cup** — the first African country to do so. "
                    "The tournament is remembered for the Vuvuzela, the ubiquitous plastic horn that became the "
                    "sound of the tournament worldwide.\n\n"
                    "Legendary players include Benni McCarthy (SA's all-time top scorer), Lucas Radebe, and "
                    "Doctor Khumalo."
                )},
            ]
        },
        {
            "messages": [
                {"role": "system", "content": S},
                {"role": "user", "content": "What is the cricket team called in South Africa?"},
                {"role": "assistant", "content": (
                    "South Africa's national cricket team is called the **Proteas**, named after the national "
                    "flower. They compete in all three formats: Tests, ODIs, and T20Is.\n\n"
                    "South Africa was banned from international cricket during apartheid (1970–1991) and returned "
                    "to the game in 1991. Notable players include Graeme Smith (long-serving captain), "
                    "AB de Villiers (widely regarded as one of the greatest batters of all time), "
                    "and Kagiso Rabada (one of the world's top fast bowlers).\n\n"
                    "The team has been competitive but has a reputation for not winning when it matters most "
                    "— they have never won a World Cup."
                )},
            ]
        },
        {
            "messages": [
                {"role": "system", "content": T},
                {"role": "user", "content": "It's Heritage Day on 24 September — what should I know about it, and what's the weather in Johannesburg?"},
                {"role": "assistant", "content": (
                    "Heritage Day (24 September) celebrates South Africa's cultural diversity and heritage. "
                    "It is also nicknamed **National Braai Day**, as South Africans of all backgrounds are "
                    "encouraged to light a fire and braai together — a unifying national tradition. "
                    "Let me check the weather in Joburg for you.\n"
                    '<tool_call>{"name": "get_weather", "arguments": {"location": "Johannesburg"}}</tool_call>'
                )},
                {"role": "user", "content": '<tool_result>{"temperature": 22, "condition": "Sunny with light breeze", "humidity": 35, "wind_kph": 12}</tool_result>'},
                {"role": "assistant", "content": (
                    "Perfect braai weather in Johannesburg — 22°C, sunny with a light breeze and low humidity. "
                    "Fire up the braai, throw on some boerewors and chops, and enjoy Heritage Day!"
                )},
            ]
        },

        # --- Wildlife ---
        {
            "messages": [
                {"role": "system", "content": S},
                {"role": "user", "content": "What are the Big Five and where can I see them in South Africa?"},
                {"role": "assistant", "content": (
                    "The Big Five are the five animals historically considered the most dangerous and difficult "
                    "to hunt on foot: **lion, leopard, elephant, white/black rhinoceros, and Cape buffalo**.\n\n"
                    "**Best places to see them in South Africa:**\n"
                    "- **Kruger National Park** (Limpopo/Mpumalanga) — the most famous, largest game reserve.\n"
                    "- **Hluhluwe-iMfolozi Park** (KwaZulu-Natal) — critically important for rhino conservation.\n"
                    "- **Addo Elephant National Park** (Eastern Cape) — particularly excellent for elephants.\n"
                    "- **Private reserves** (Sabi Sand, Timbavati) — higher sighting rates with expert guides.\n\n"
                    "May–September is the best time for sightings when bush is dry and sparse."
                )},
            ]
        },
        {
            "messages": [
                {"role": "system", "content": S},
                {"role": "user", "content": "How serious is rhino poaching in South Africa?"},
                {"role": "assistant", "content": (
                    "Rhino poaching is one of South Africa's most serious conservation crises. South Africa is "
                    "home to approximately 80% of the world's remaining white and black rhinos.\n\n"
                    "**The scale:** At the peak in 2014, over 1,200 rhinos were killed in a single year. "
                    "Demand is driven by illegal markets — primarily in Vietnam and China — where rhino horn "
                    "is falsely believed to have medicinal properties. Horn sells for more than gold per kilogram.\n\n"
                    "**Responses:** Anti-poaching units (some military-grade), dehorning programmes (removing "
                    "the horn to make rhinos less valuable targets), private reserve security, and international "
                    "law enforcement cooperation. Kruger National Park remains the hardest-hit area."
                )},
            ]
        },
        {
            "messages": [
                {"role": "system", "content": S},
                {"role": "user", "content": "When is the best time to visit Kruger National Park?"},
                {"role": "assistant", "content": (
                    "**Best time: May to September (dry winter season)**\n\n"
                    "- Vegetation is thin and sparse — easier to spot animals through the bush.\n"
                    "- Animals congregate around rivers and water holes, making sightings predictable.\n"
                    "- Temperatures are pleasant (15–25°C), with cold nights.\n"
                    "- Little rain, fewer mosquitoes, lower malaria risk.\n\n"
                    "**October to April (wet summer):**\n"
                    "- Lush, green park — beautiful but harder to spot game.\n"
                    "- Many animals give birth (great for predator activity).\n"
                    "- Hot and humid, higher malaria risk, afternoon thunderstorms.\n"
                    "- Lower accommodation rates.\n\n"
                    "Peak school holiday months (June/July and December) get busy — book rest camp "
                    "accommodation well in advance through SANParks."
                )},
            ]
        },
        {
            "messages": [
                {"role": "system", "content": S},
                {"role": "user", "content": "I heard South Africa is famous for shark cage diving. Where can I do this?"},
                {"role": "assistant", "content": (
                    "Yes — Gansbaai, a small fishing town about 160 km east of Cape Town in the Western Cape, "
                    "is considered the **Great White Shark capital of the world**.\n\n"
                    "The nutrient-rich waters between Gansbaai and Dyer Island create what is known as "
                    "**Shark Alley**, where large numbers of Great White Sharks congregate to hunt Cape fur "
                    "seals on the island.\n\n"
                    "Cage diving operators lower a steel cage into the water while sharks are attracted with "
                    "bait. You don't need diving experience — you simply hold your breath and view through "
                    "the cage. The experience is available year-round, with the best conditions typically "
                    "in winter (June–August) when sharks are most active."
                )},
            ]
        },

        # --- Cultural Practices ---
        {
            "messages": [
                {"role": "system", "content": S},
                {"role": "user", "content": "What is lobola?"},
                {"role": "assistant", "content": (
                    "Lobola (also called bridewealth) is a traditional practice common among many Southern "
                    "African cultures, including Zulu, Xhosa, Sotho, Tswana, and Ndebele peoples.\n\n"
                    "It involves a negotiation between the families of the bride and groom, where the groom's "
                    "family offers cattle (or increasingly, their cash equivalent) to the bride's family. "
                    "The number and value are negotiated — typically 8–15 cows.\n\n"
                    "**Common misconceptions:** Lobola is NOT 'buying' a wife. It is about:\n"
                    "- Strengthening bonds between two families.\n"
                    "- Expressing gratitude to the bride's family for raising her.\n"
                    "- Giving the marriage formal social recognition within the community.\n\n"
                    "Without lobola, many communities do not consider a couple traditionally married, "
                    "even if they have a civil or church ceremony."
                )},
            ]
        },
        {
            "messages": [
                {"role": "system", "content": S},
                {"role": "user", "content": "What is a sangoma?"},
                {"role": "assistant", "content": (
                    "A sangoma is a traditional healer and spiritual practitioner in Nguni cultures "
                    "(Zulu, Xhosa, Swati, Ndebele). The term is often incorrectly used interchangeably "
                    "with 'witch doctor' — but this is a misnomer with negative colonial connotations.\n\n"
                    "**A sangoma's role:**\n"
                    "- Acts as a medium between the living and ancestral spirits (amadlozi in Zulu).\n"
                    "- Diagnoses illness by throwing bones (hakata), reading patterns, or entering trance states.\n"
                    "- Prescribes herbal remedies and spiritual rituals.\n"
                    "- Guides community members through life transitions (birth, death, illness, disputes).\n\n"
                    "**Sangoma vs Inyanga:** An inyanga is more of an herbalist — focused on plant medicines "
                    "without the spiritual divination role of a sangoma.\n\n"
                    "Millions of South Africans consult sangomas alongside or instead of Western medicine."
                )},
            ]
        },
        {
            "messages": [
                {"role": "system", "content": S},
                {"role": "user", "content": "What is the significance of beadwork in Zulu culture?"},
                {"role": "assistant", "content": (
                    "In Zulu culture, beadwork (izigqizo) is a sophisticated communication system, not just "
                    "decoration. Zulu beadwork uses colour, pattern, and shape to encode messages — "
                    "traditionally made by women and worn by both men and women.\n\n"
                    "**Colour meanings (can vary by region and family):**\n"
                    "- White — purity, love, spiritual matters\n"
                    "- Red — strong emotion (love or anger depending on context)\n"
                    "- Blue — faithfulness, request\n"
                    "- Black — marriage, sorrow\n"
                    "- Yellow/Gold — wealth, fertility\n"
                    "- Green — jealousy or domestic bliss\n\n"
                    "Zulu love letters (ucu) are beaded strips traditionally sent by young women to men they "
                    "admire, encoding a message readable only by the intended recipient who knows the code. "
                    "This tradition continues today alongside tourism-focused beadwork art."
                )},
            ]
        },

        # --- Government Services ---
        {
            "messages": [
                {"role": "system", "content": S},
                {"role": "user", "content": "How do I apply for a South African ID?"},
                {"role": "assistant", "content": (
                    "To apply for a South African Smart ID Card through the Department of Home Affairs (DHA):\n\n"
                    "**Requirements:**\n"
                    "- Certified copy of your birth certificate (or original for first-time applicants).\n"
                    "- If you already have a green barcoded ID book, bring it.\n\n"
                    "**Process:**\n"
                    "1. Book an appointment via the DHA website (dha.gov.za) or eHome Affairs app — "
                    "   smart ID cards require appointments at a DHA office or participating bank.\n"
                    "2. Attend the appointment. Biometric data (fingerprints, photo) will be captured.\n"
                    "3. Pay the fee (R140 for a first Smart ID; replacements are also R140).\n"
                    "4. Collection takes 2–8 weeks depending on your office.\n\n"
                    "**Note:** South African citizens qualify for a Smart ID at age 16. The old green barcoded "
                    "ID book remains legally valid but the Smart ID is now preferred."
                )},
            ]
        },
        {
            "messages": [
                {"role": "system", "content": S},
                {"role": "user", "content": "How do I report a crime to the South African Police Service?"},
                {"role": "assistant", "content": (
                    "To report a crime to the SAPS (South African Police Service):\n\n"
                    "**In an emergency:** Call **10111** (free from any phone) or **112** (mobile emergency number).\n\n"
                    "**For non-emergency reports:**\n"
                    "1. Go to the nearest police station and ask for the charge office.\n"
                    "2. Make a statement to the duty officer. Be as detailed as possible — date, time, location, "
                    "   description of suspects and any property involved.\n"
                    "3. You will receive a **case number** (CAS number) — keep this, it is your reference for "
                    "   following up on your case.\n\n"
                    "**Online/digital options:**\n"
                    "- The **MySAPS app** allows you to report certain crimes online.\n"
                    "- You can also report crimes anonymously via the Crime Stop hotline: **0860 010 111**.\n\n"
                    "You have the right to lay a complaint in any of South Africa's 11 official languages."
                )},
            ]
        },
        {
            "messages": [
                {"role": "system", "content": S},
                {"role": "user", "content": "How do I apply for a South African passport?"},
                {"role": "assistant", "content": (
                    "To apply for a South African passport:\n\n"
                    "**Requirements:**\n"
                    "- Original green barcoded ID book or Smart ID card.\n"
                    "- Certified copy of your birth certificate (first-time applicants).\n"
                    "- Completed DHA-73 application form.\n"
                    "- Two recent passport-sized photos (35mm x 45mm, white background).\n\n"
                    "**Process:**\n"
                    "1. Book an appointment at a DHA office (dha.gov.za or eHome Affairs portal).\n"
                    "2. Attend the appointment with original documents.\n"
                    "3. Pay the fee: R400 for a 32-page passport; R800 for a 48-page business passport.\n"
                    "4. Processing takes 4–10 weeks at a DHA office. Express passports (10 working days) "
                    "   are available at Johannesburg international airport DHA office at higher cost.\n\n"
                    "**Validity:** South African passports are valid for 10 years for adults.\n\n"
                    "Tip: Apply well in advance of any planned travel — processing times are often longer "
                    "than advertised."
                )},
            ]
        },
    ]


# ---------------------------------------------------------------------------
# Translation utilities
# ---------------------------------------------------------------------------

def translate_text(text: str, target_lang: str, sleep_sec: float = 0.5) -> str | None:
    from deep_translator import GoogleTranslator
    from deep_translator.exceptions import LanguageNotSupportedException

    time.sleep(sleep_sec)
    try:
        result = GoogleTranslator(source="en", target=target_lang).translate(text)
        return result if result else None
    except LanguageNotSupportedException:
        raise
    except Exception as exc:
        if "429" in str(exc) or "Too Many Requests" in str(exc).replace("_", " "):
            logging.warning(f"Rate limited for lang={target_lang}, backing off 30s ...")
            time.sleep(30)
            try:
                return GoogleTranslator(source="en", target=target_lang).translate(text)
            except Exception:
                return None
        logging.warning(f"Translation error for lang={target_lang}: {exc}")
        return None


def translate_content_preserving_tags(content: str, target_lang: str) -> str | None:
    parts = _TAG_PATTERN.split(content)
    translated_parts: list[str] = []
    for part in parts:
        if _TAG_PATTERN.match(part):
            translated_parts.append(part)
        elif part.strip():
            result = translate_text(part, target_lang)
            if result is None:
                return None
            translated_parts.append(result)
        else:
            translated_parts.append(part)
    return "".join(translated_parts)


def translate_messages(messages: list[dict], target_lang: str) -> list[dict] | None:
    translated: list[dict] = []
    for msg in messages:
        content = translate_content_preserving_tags(msg["content"], target_lang)
        if content is None:
            return None
        translated.append({"role": msg["role"], "content": content})
    return translated


# ---------------------------------------------------------------------------
# Dataset assembly
# ---------------------------------------------------------------------------

def generate_language_split(
    english_examples: list[dict],
    lang_name: str,
    lang_code: str,
    translate: bool,
    id_offset: int,
) -> list[dict]:
    from deep_translator.exceptions import LanguageNotSupportedException

    records: list[dict] = []

    if not translate:
        for i, ex in enumerate(english_examples):
            records.append({
                "id": id_offset + i,
                "language": lang_name,
                "language_code": lang_code,
                "messages": copy.deepcopy(ex["messages"]),
            })
        return records

    # Probe with first example to detect unsupported languages early
    if lang_code in CONFIRMED_UNSUPPORTED:
        logging.warning(f"Skipping {lang_name} ({lang_code}) — previously confirmed unsupported.")
        return []

    try:
        probe = translate_messages(english_examples[0]["messages"], lang_code)
    except LanguageNotSupportedException:
        logging.warning(f"Language not supported by Google Translate: {lang_name} ({lang_code}). Skipping.")
        CONFIRMED_UNSUPPORTED.add(lang_code)
        return []

    if probe is None:
        logging.warning(f"Probe translation failed for {lang_name} ({lang_code}). Skipping.")
        return []

    records.append({
        "id": id_offset,
        "language": lang_name,
        "language_code": lang_code,
        "messages": probe,
    })

    for i, ex in enumerate(english_examples[1:], start=1):
        if lang_code in CONFIRMED_UNSUPPORTED:
            break
        try:
            translated = translate_messages(ex["messages"], lang_code)
        except LanguageNotSupportedException:
            logging.warning(f"Language not supported: {lang_name} ({lang_code}). Stopping mid-split.")
            CONFIRMED_UNSUPPORTED.add(lang_code)
            break
        if translated is None:
            logging.warning(f"  Skipping example {i} for {lang_name} — translation returned None.")
            continue
        records.append({
            "id": id_offset + i,
            "language": lang_name,
            "language_code": lang_code,
            "messages": translated,
        })

    return records


def save_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    logging.info(f"  Saved {len(records)} records → {path}")


def build_dataset_dict(all_splits: dict[str, list[dict]]):
    from datasets import Dataset, DatasetDict, Features, Value, Sequence

    features = Features({
        "id": Value("int32"),
        "language": Value("string"),
        "language_code": Value("string"),
        "messages": [{"role": Value("string"), "content": Value("string")}],
    })

    splits: dict = {}
    for lang_name, records in all_splits.items():
        if records:
            splits[lang_name] = Dataset.from_list(records, features=features)

    return DatasetDict(splits)


def push_to_hub(dataset, repo_id: str, token: str) -> None:
    print(f"\nPushing dataset to https://huggingface.co/datasets/{repo_id} ...")
    dataset.push_to_hub(repo_id, token=token, private=False)
    print(f"Done. Dataset available at: https://huggingface.co/datasets/{repo_id}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate SA 11-language dataset and push to Hugging Face.")
    p.add_argument("--username", default=None, help="Hugging Face username (required unless --skip-upload)")
    p.add_argument("--repo-name", default="mzansilm-sa11", help="HF dataset repo name (default: mzansilm-sa11)")
    p.add_argument("--token", default=None, help="HF token (falls back to HF_TOKEN env var)")
    p.add_argument("--output-dir", default="./data", help="Directory for local JSONL files (default: ./data)")
    p.add_argument("--skip-upload", action="store_true", help="Save JSONL files locally only, skip HF push")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)

    english_examples = get_english_examples()
    logging.info(f"Loaded {len(english_examples)} English source examples.")

    all_splits: dict[str, list[dict]] = {}
    global_id = 0

    for lang_name, lang_code in LANGUAGES.items():
        is_source = lang_code == "en"
        print(f"\n[{lang_name}] ({lang_code}){' — source, no translation' if is_source else ''}")

        records = generate_language_split(
            english_examples,
            lang_name=lang_name,
            lang_code=lang_code,
            translate=not is_source,
            id_offset=global_id,
        )

        all_splits[lang_name] = records
        if records:
            save_jsonl(records, output_dir / f"{lang_name}.jsonl")
        else:
            print(f"  No records for {lang_name} — skipped.")

        global_id += len(records)

    total = sum(len(v) for v in all_splits.values())
    print(f"\nTotal records across all languages: {total}")

    if args.skip_upload:
        print(f"Skipping Hugging Face upload (--skip-upload). Files in: {output_dir.resolve()}")
        return

    if not args.username:
        sys.exit("Error: --username is required for upload. Use --skip-upload for local-only mode.")

    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("Error: provide --token or set the HF_TOKEN environment variable.")

    repo_id = f"{args.username}/{args.repo_name}"
    dataset = build_dataset_dict(all_splits)
    push_to_hub(dataset, repo_id, token)


if __name__ == "__main__":
    main()
