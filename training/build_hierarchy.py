"""Broad labels for the training data: every fine label of CLINC150, Banking77, MASSIVE and the four Bitext sets
gets a broad label (its domain, scenario or category). Training then shows the same text with fine options in some
lists and broad options in others, so the model learns that a concrete request belongs to a broad category.

CLINC150 domains follow the ten domains of the CLINC paper (15 intents each). Banking77 has no hierarchy; the
groups below are ours. MASSIVE uses its own scenario (the first word of the intent). Bitext uses its own category
column, with the single-intent insurance categories merged. Broad labels never equal a test label (checked).

    python build_hierarchy.py  ->  data/coarse_map.json   {source: {fine label: broad label}}
"""
import json
import os
from collections import defaultdict


def _json_load(*args, **kw):
    with open(*args, **kw) as f:
        return json.load(f)


def _jsonl(*args, **kw):
    with open(*args, **kw) as f:
        return [json.loads(line) for line in f]

HERE = os.path.dirname(os.path.abspath(__file__))

CLINC = {
    "Bank accounts and payments": "transfer, transactions, balance, freeze account, pay bill, bill balance, bill due, "
        "interest rate, routing, min payment, order checks, pin change, report fraud, account blocked, spending history",
    "Credit cards": "credit score, report lost card, credit limit, rewards balance, new card, application status, "
        "card declined, international fees, apr, redeem rewards, credit limit change, damaged card, "
        "replacement card duration, improve credit score, expiration date",
    "Food, cooking and restaurants": "recipe, restaurant reviews, calories, nutrition info, restaurant suggestion, "
        "ingredients list, ingredient substitution, cook time, food last, meal suggestion, restaurant reservation, "
        "confirm reservation, how busy, cancel reservation, accept reservations",
    "Home, lists and music": "shopping list, shopping list update, next song, play music, update playlist, todo list, "
        "todo list update, calendar, calendar update, what song, order, order status, reminder, reminder update, "
        "smart home",
    "Cars and commuting": "traffic, directions, gas, gas type, distance, current location, mpg, oil change when, "
        "oil change how, jump start, uber, schedule maintenance, last maintenance, tire pressure, tire change",
    "Travel": "book flight, book hotel, car rental, travel suggestion, travel alert, travel notification, carry on, "
        "timezone, vaccines, translate, flight status, international visa, lost luggage, plug type, exchange rate",
    "Everyday utilities": "time, alarm, share location, find phone, weather, text, spelling, make call, timer, date, "
        "calculator, measurement conversion, flip coin, roll dice, definition",
    "Work, pay and benefits": "direct deposit, pto request, taxes, payday, w2, pto balance, pto request status, "
        "next holiday, insurance, insurance change, schedule meeting, pto used, meeting schedule, rollover 401k, income",
    "Small talk with the assistant": "greeting, goodbye, tell joke, where are you from, how old are you, "
        "what is your name, who made you, thank you, what can i ask you, what are your hobbies, do you have pets, "
        "are you a bot, meaning of life, who do you work for, fun fact",
    "Assistant settings and replies": "change ai name, change user name, cancel, user name, reset settings, "
        "whisper mode, repeat, no, yes, maybe, change language, change accent, change volume, change speed, "
        "sync device",
}

BANKING77 = {
    "Getting a card": "activate my card, card arrival, card delivery estimate, get physical card, order physical card, "
        "getting spare card, getting virtual card, get disposable virtual card, card linking, card about to expire, "
        "disposable card limits",
    "Card not working or lost": "card not working, virtual card not working, contactless not working, card swallowed, "
        "lost or stolen card, compromised card, pin blocked, change pin, declined card payment",
    "Card payments and charges": "card payment fee charged, card payment not recognised, card payment wrong exchange rate, "
        "pending card payment, reverted card payment, transaction charged twice, extra charge on statement, "
        "card acceptance, apple pay or google pay, visa or mastercard",
    "Cash and ATMs": "atm support, cash withdrawal charge, cash withdrawal not recognised, declined cash withdrawal, "
        "pending cash withdrawal, wrong amount of cash received, wrong exchange rate for cash withdrawal",
    "Topping up the account": "automatic top up, pending top up, top up by bank transfer charge, top up by card charge, "
        "top up by cash or cheque, top up failed, top up limits, top up reverted, topping up by card, verify top up",
    "Transfers and incoming money": "balance not updated after bank transfer, balance not updated after cheque or cash deposit, "
        "beneficiary not allowed, cancel transfer, declined transfer, failed transfer, pending transfer, "
        "receiving money, transfer fee charged, transfer into account, transfer not received by recipient, "
        "transfer timing, direct debit payment not recognised",
    "Currencies and exchange": "exchange charge, exchange rate, exchange via app, fiat currency support, "
        "supported cards and currencies, country support",
    "Identity and account details": "age limit, edit personal details, passcode forgotten, lost or stolen phone, "
        "terminate account, unable to verify identity, verify my identity, verify source of funds, why verify identity",
    "Refunds": "refund not showing up, request refund",
}

MASSIVE_EN = {
    "alarm": "Alarms", "audio": "Speaker volume", "calendar": "Calendar and events", "cooking": "Cooking and recipes",
    "datetime": "Dates and times", "email": "Email and contacts", "general": "Chit-chat and assistant commands",
    "iot": "Smart home devices", "lists": "Lists and to-dos", "music": "Music likes and settings",
    "news": "News and headlines", "play": "Playing media", "qa": "General knowledge questions",
    "recommendation": "Recommendations", "social": "Social networks", "takeaway": "Takeaway food",
    "transport": "Transport and tickets", "weather": "Weather forecasts",
}
MASSIVE_ES = {
    "alarm": "Alarmas", "audio": "Volumen del altavoz", "calendar": "Calendario y eventos", "cooking": "Cocina y recetas",
    "datetime": "Fechas y horas", "email": "Correo y contactos", "general": "Charla y órdenes al asistente",
    "iot": "Dispositivos del hogar", "lists": "Listas y tareas", "music": "Gustos y ajustes de música",
    "news": "Noticias", "play": "Reproducir contenido", "qa": "Preguntas de cultura general",
    "recommendation": "Recomendaciones", "social": "Redes sociales", "takeaway": "Comida para llevar",
    "transport": "Transporte y billetes", "weather": "Previsión del tiempo",
}

BITEXT = {
    "bitext_support": {"ORDER": "Orders", "SHIPPING": "Shipping address", "CANCEL": "Cancellation fees",
                       "INVOICE": "Invoices", "PAYMENT": "Payment methods and problems", "REFUND": "Refunds and returns",
                       "FEEDBACK": "Complaints and reviews", "CONTACT": "Talking to a person",
                       "ACCOUNT": "Customer account", "DELIVERY": "Delivery times and options",
                       "SUBSCRIPTION": "Newsletter"},
    "bitext_banking": {"CARD": "Cards", "LOAN": "Loans and mortgages", "TRANSFER": "Transfers", "FEES": "Bank fees",
                       "ACCOUNT": "Accounts", "CONTACT": "Talking to a person", "ATM": "Cash machine problems",
                       "FIND": "Branches and cash machines nearby", "PASSWORD": "Passwords"},
    "bitext_insurance": {"CLAIMS": "Claims and settlements", "COMPLAINTS": "Complaints and disputes",
                         "CONTACT": "Talking to a person", "COVERAGE": "Coverage changes",
                         "ENROLLMENT": "Buying or cancelling a policy", "GENERAL_INFORMATION": "Insurance products",
                         "AUTO_INSURANCE": "Insurance products", "HEALTH_INSURANCE": "Insurance products",
                         "HOME_INSURANCE": "Insurance products", "LIFE_INSURANCE": "Insurance products",
                         "PET_INSURANCE": "Insurance products", "TRAVEL_INSURANCE": "Insurance products",
                         "INCIDENTS": "Reporting an incident", "PAYMENT": "Premium payments",
                         "POLICY": "Personal details on the policy", "QUOTE": "Quotes and rates",
                         "RENEW": "Policy renewal"},
    "bitext_telco": {"BILLING": "Bills", "COMPLAINTS": "Faults and compensation", "CONSUMPTION": "Data usage",
                     "CONTACT": "Talking to a person", "PAYMENT": "Payments", "SERVICES": "Lines and services",
                     "SUBSCRIPTION": "Plans and contracts"},
}
BITEXT_REPO = {"bitext_support": "bitext/Bitext-customer-support-llm-chatbot-training-dataset",
               "bitext_banking": "bitext/Bitext-retail-banking-llm-chatbot-training-dataset",
               "bitext_insurance": "bitext/Bitext-insurance-llm-chatbot-training-dataset",
               "bitext_telco": "bitext/Bitext-telco-llm-chatbot-training-dataset"}


def norm(label):  # same as prep_data.norm
    import re
    s = label.replace("+", " and ").replace("?", "")
    s = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", s)
    s = s.replace("_", " ").replace("-", " ").lower()
    return re.sub(r"\s+", " ", s).strip()


def invert(groups):
    out = {}
    for broad, fines in groups.items():
        for f in fines.split(","):
            f = f.strip()
            assert f not in out, f"{f} in two groups"
            out[f] = broad
    return out


def main():
    labels = defaultdict(set)
    for r in _jsonl(os.path.join(HERE, "data", "train.jsonl"), encoding="utf-8"):
        labels[r["source"]].add(r["label"])
    cmap = {"clinc150": invert(CLINC), "banking77": invert(BANKING77)}
    cmap["massive_en"] = {l: MASSIVE_EN[l.split()[0]] for l in labels["massive_en"]}
    cmap["massive_es"] = {l: MASSIVE_ES[l.split()[0]] for l in labels["massive_es"]}
    from datasets import load_dataset
    for src, repo in BITEXT_REPO.items():
        d = load_dataset(repo)["train"]
        m = {}
        for intent, cat in zip(d["intent"], d["category"]):
            m[norm(intent)] = BITEXT[src][cat]
        cmap[src] = m
    forbidden = set(_json_load(os.path.join(HERE, "forbidden_labels.json")))
    for src, m in cmap.items():
        missing = labels[src] - set(m)
        extra = set(m) - labels[src]
        assert not missing, f"{src}: fine labels without a broad label: {sorted(missing)}"
        assert not extra, f"{src}: unknown fine labels in the map: {sorted(extra)}"
        bad = {b for b in m.values() if b.lower() in forbidden}
        assert not bad, f"{src}: broad labels equal to test labels: {bad}"
        print(f"{src:18s} {len(m):3d} fine -> {len(set(m.values())):2d} broad")
    with open(os.path.join(HERE, "data", "coarse_map.json"), "w", encoding="utf-8") as f:
        json.dump(cmap, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
