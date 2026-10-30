#!/usr/bin/env python3
"""
PromptShield Benchmark Dataset Expander (105 -> 250 Prompts)
=============================================================
Adds 145 realistic, enterprise-grade, and diverse prompts to reach
exactly 250 prompts in ground_truth.json.

Ensures:
1. Exact character-level offset calculation: text[start:end] == entity['text']
2. Diverse global names (Indian, European, American, Asian, Hispanic, Middle Eastern)
3. Diverse technical credentials, Luhn-valid credit cards, IBANs, IP addresses, JWT tokens
4. Negative controls (zero-PII prompts) to evaluate precision robustness
5. Strict offset verification before saving
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

BENCHMARK_DIR = Path(__file__).resolve().parent
GT_FILE = BENCHMARK_DIR / "ground_truth.json"


def build_prompt_with_entities(template: str, entities_to_place: List[Tuple[str, str, str]]) -> Dict:
    """
    Constructs a prompt and computes exact [start, end) offsets for each entity.
    entities_to_place: List of (placeholder_key, entity_text, label)
    template: Text containing {placeholder_key} markers.
    """
    # Replace placeholders sequentially while tracking exact offsets
    current_text = template
    computed_entities = []

    for placeholder_key, entity_val, entity_label in entities_to_place:
        token = "{" + placeholder_key + "}"
        pos = current_text.find(token)
        if pos == -1:
            raise ValueError(f"Placeholder {token} not found in template: {current_text}")
        
        current_text = current_text[:pos] + entity_val + current_text[pos + len(token):]
        computed_entities.append({
            "text": entity_val,
            "start": pos,
            "end": pos + len(entity_val),
            "label": entity_label,
            "_key": placeholder_key,
        })

    # Sort entities by start offset
    computed_entities.sort(key=lambda e: e["start"])

    # Double-check character offsets against the final rendered prompt
    clean_entities = []
    for e in computed_entities:
        actual = current_text[e["start"]:e["end"]]
        if actual != e["text"]:
            raise ValueError(f"Offset mismatch: expected '{e['text']}' got '{actual}' in prompt: {current_text}")
        clean_entities.append({
            "text": e["text"],
            "start": e["start"],
            "end": e["end"],
            "label": e["label"]
        })

    return {
        "text": current_text,
        "entities": clean_entities
    }


def generate_145_new_prompts(start_id: int = 106) -> List[Dict]:
    raw_specs = [
        # --- 1. Cloud & API Credentials ---
        (
            "Rotate the production OpenAI key {k1} and update the AWS secret {k2} before deploying to {loc}.",
            [("k1", "sk-live-9928374610293847561029384756", "API_KEY"),
             ("k2", "AKIAIOSFODNN7EXAMPLE", "API_KEY"),
             ("loc", "Frankfurt", "LOCATION")]
        ),
        (
            "Use the Google Cloud API key {k1} to authenticate the maps service for our {org} client.",
            [("k1", "AIzaSyD-9xK2Lm3No4Pq5Rs6Tu7Vw8Xy9Za0Bc", "API_KEY"),
             ("org", "Siemens", "ORGANIZATION")]
        ),
        (
            "The authorization header should contain the bearer token {tok} for the API gateway.",
            [("tok", "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozG42iP_k8xT", "ACCESS_TOKEN")]
        ),
        (
            "GitHub personal access token {pat} was revoked by security administrator {name}.",
            [("pat", "ghp_9876543210abcdefghijklmnopqrstuvwx", "API_KEY"),
             ("name", "Marcus Vance", "PERSON")]
        ),
        (
            "Database connection URI is {conn} for the billing replica.",
            [("conn", "postgresql://dbadmin:P@ssw0rd2024!@10.0.12.44:5432/finance", "PASSWORD")]
        ),
        (
            "Please reset the admin password to {pwd} on server {ip}.",
            [("pwd", "Sapphire#9834@Secure", "PASSWORD"),
             ("ip", "192.168.100.25", "IP_ADDRESS")]
        ),
        (
            "Temporary access pin for locker room is {pin} given to contractor {name}.",
            [("pin", "9482", "PASSWORD"),
             ("name", "David Chen", "PERSON")]
        ),
        (
            "Production Redis cluster requires password {pwd} on port 6379.",
            [("pwd", "RedisAuthPass991#", "PASSWORD")]
        ),

        # --- 2. Banking, Financial Identifiers & Luhn Credit Cards ---
        (
            "Wire the milestone fee of 45,000 EUR to IBAN {iban} at {org} in {loc}.",
            [("iban", "DE89370400440532013000", "BANK_ACCOUNT"),
             ("org", "Deutsche Bank", "ORGANIZATION"),
             ("loc", "Munich", "LOCATION")]
        ),
        (
            "Customer account number {acc} received a direct deposit on {dt}.",
            [("acc", "987123456789", "BANK_ACCOUNT"),
             ("dt", "24/10/2024", "DATE")]
        ),
        (
            "The customer used Visa card {cc} to pay for flight booking {ord}.",
            [("cc", "4532015112830366", "CREDIT_CARD"),
             ("ord", "ORD-88210", "ORDER_ID")]
        ),
        (
            "Charged Mastercard {cc} for annual subscription order {ord}.",
            [("cc", "5425233430109903", "CREDIT_CARD"),
             ("ord", "ORD-44910", "ORDER_ID")]
        ),
        (
            "American Express payment {cc} was declined for invoice {ord}.",
            [("cc", "378282246310005", "CREDIT_CARD"),
             ("ord", "ORD-11029", "ORDER_ID")]
        ),
        (
            "Refund the deposit of 250 USD to account {acc} belonging to {name}.",
            [("acc", "112233445566", "BANK_ACCOUNT"),
             ("name", "Elena Rostova", "PERSON")]
        ),
        (
            "Beneficiary IBAN is {iban} registered under {org}.",
            [("iban", "GB29NWBK60161331926819", "BANK_ACCOUNT"),
             ("org", "NatWest Markets", "ORGANIZATION")]
        ),

        # --- 3. Enterprise Ticketing, Orders & Customer Support ---
        (
            "Incident ticket {tkt} was assigned to senior engineer {name} at {org}.",
            [("tkt", "TKT-44912", "TICKET_ID"),
             ("name", "Siddharth Rao", "PERSON"),
             ("org", "Wipro", "ORGANIZATION")]
        ),
        (
            "Customer {name} (ID: {cid}) inquired about shipment {ord} heading to {loc}.",
            [("name", "Amina Al-Mansoor", "PERSON"),
             ("cid", "CUST-99412", "CUSTOMER_ID"),
             ("ord", "ORD-77194", "ORDER_ID"),
             ("loc", "Dubai", "LOCATION")]
        ),
        (
            "Support ticket {tkt} created by {email} regarding billing error on {dt}.",
            [("tkt", "TKT-88319", "TICKET_ID"),
             ("email", "support.client@acme.org", "EMAIL"),
             ("dt", "19/09/2024", "DATE")]
        ),
        (
            "Order {ord} placed by customer {cid} was delivered to {addr}.",
            [("ord", "ORD-20391", "ORDER_ID"),
             ("cid", "CUST-33190", "CUSTOMER_ID"),
             ("addr", "742 Evergreen Terrace, Springfield, OR 97477", "ADDRESS")]
        ),
        (
            "Please check ticket {tkt} opened by employee {cid} on {dt}.",
            [("tkt", "TKT-10928", "TICKET_ID"),
             ("cid", "CUST-55102", "CUSTOMER_ID"),
             ("dt", "03/11/2024", "DATE")]
        ),
        (
            "Dispatch technician {name} to resolve order {ord} at {loc}.",
            [("name", "Carlos Mendez", "PERSON"),
             ("ord", "ORD-66291", "ORDER_ID"),
             ("loc", "Madrid", "LOCATION")]
        ),
        (
            "Customer {name} registered with email {email} under client account {cid}.",
            [("name", "Fatima Zahra", "PERSON"),
             ("email", "fatima.zahra@casablanca.ma", "EMAIL"),
             ("cid", "CUST-88321", "CUSTOMER_ID")]
        ),

        # --- 4. Personal Names, Contacts & Global Identities ---
        (
            "Dr. {name} can be reached at {email} or mobile {ph}.",
            [("name", "Sunita Deshmukh", "PERSON"),
             ("email", "s.deshmukh@aiims.edu.in", "EMAIL"),
             ("ph", "+91 9988776655", "PHONE")]
        ),
        (
            "Contact project lead {name} in {loc} at {ph}.",
            [("name", "Jean-Pierre Dubois", "PERSON"),
             ("loc", "Lyon", "LOCATION"),
             ("ph", "+33 4 72 00 00 00", "PHONE")]
        ),
        (
            "Consultant {name} from {org} submitted the architectural review on {dt}.",
            [("name", "Kenji Takahashi", "PERSON"),
             ("org", "Sony", "ORGANIZATION"),
             ("dt", "14 October 2024", "DATE")]
        ),
        (
            "Send meeting minutes to {name} at {email} before end of day.",
            [("name", "Hannah Schmidt", "PERSON"),
             ("email", "hannah.schmidt@berlin-tech.de", "EMAIL")]
        ),
        (
            "Senior architect {name} relocated from {loc1} to {loc2} for the new {org} project.",
            [("name", "Vikramaditya Singhania", "PERSON"),
             ("loc1", "Delhi", "LOCATION"),
             ("loc2", "Singapore", "LOCATION"),
             ("org", "Temasek", "ORGANIZATION")]
        ),
        (
            "Emergency contact for patient {name} is {ph}.",
            [("name", "Maria Santos", "PERSON"),
             ("ph", "+55 11 98765-4321", "PHONE")]
        ),
        (
            "Lead researcher {name} published a paper on neural privacy on {dt}.",
            [("name", "Li Wei", "PERSON"),
             ("dt", "28 August 2024", "DATE")]
        ),
        (
            "Notify HR specialist {name} at {email} regarding the job opening in {loc}.",
            [("name", "Tariq Mahmood", "PERSON"),
             ("email", "tariq.m@dubaicorp.ae", "EMAIL"),
             ("loc", "Abu Dhabi", "LOCATION")]
        ),

        # --- 5. Network Architecture, IPs & Infrastructure ---
        (
            "The external DNS gateway is hosted at {ip} with backup at {ip2}.",
            [("ip", "198.51.100.14", "IP_ADDRESS"),
             ("ip2", "203.0.113.89", "IP_ADDRESS")]
        ),
        (
            "Ingress traffic is routed through IPv6 node {ip} in data center {loc}.",
            [("ip", "2001:0db8:3333:4444:5555:6666:7777:8888", "IP_ADDRESS"),
             ("loc", "Amsterdam", "LOCATION")]
        ),
        (
            "Internal monitoring agent connected from {ip} to server in {loc}.",
            [("ip", "10.244.15.8", "IP_ADDRESS"),
             ("loc", "Stockholm", "LOCATION")]
        ),
        (
            "Firewall rule blocks inbound packets from subnet {ip} to protect {org} systems.",
            [("ip", "172.16.45.100", "IP_ADDRESS"),
             ("org", "Nordea Bank", "ORGANIZATION")]
        ),
        (
            "VPC gateway interface IP {ip} is managed by network lead {name}.",
            [("ip", "192.168.220.1", "IP_ADDRESS"),
             ("name", "Alexei Volkov", "PERSON")]
        ),

        # --- 6. Addresses & Physical Residences ---
        (
            "Ship the prototype hardware to {addr} attention of {name}.",
            [("addr", "10 Downing Street, London, SW1A 2AA", "ADDRESS"),
             ("name", "Oliver Smith", "PERSON")]
        ),
        (
            "The headquarters of the subsidiary is located at {addr}.",
            [("addr", "221B Baker Street, Marylebone, London NW1 6XE", "ADDRESS")]
        ),
        (
            "Deliver parcel to {addr} for recipient {name}.",
            [("addr", "350 5th Avenue, New York, NY 10118", "ADDRESS"),
             ("name", "Rachel Green", "PERSON")]
        ),
        (
            "Billing address on record is {addr} for customer {cid}.",
            [("addr", "12 Rue de la Paix, 75002 Paris, France", "ADDRESS"),
             ("cid", "CUST-44109", "CUSTOMER_ID")]
        ),
        (
            "Branch office address: {addr} in {loc}.",
            [("addr", "50 Raffles Place, Singapore 048623", "ADDRESS"),
             ("loc", "Singapore", "LOCATION")]
        ),

        # --- 7. Multi-Entity Complex Prompts ---
        (
            "On {dt}, {name1} from {org1} met with {name2} from {org2} in {loc} to sign agreement {ord}.",
            [("dt", "15 November 2024", "DATE"),
             ("name1", "Rajesh Pillai", "PERSON"),
             ("org1", "TCS", "ORGANIZATION"),
             ("name2", "Ingrid Bergman", "PERSON"),
             ("org2", "Ericsson", "ORGANIZATION"),
             ("loc", "Stockholm", "LOCATION"),
             ("ord", "ORD-99012", "ORDER_ID")]
        ),
        (
            "Employee {name} submitted expense report {ord} on {dt} for travel to {loc}. Contact: {email}.",
            [("name", "Kwame Mensah", "PERSON"),
             ("ord", "ORD-12840", "ORDER_ID"),
             ("dt", "05/10/2024", "DATE"),
             ("loc", "Accra", "LOCATION"),
             ("email", "k.mensah@ghana-tech.com", "EMAIL")]
        ),
        (
            "Customer {name} (phone: {ph}) reported fraudulent charge {cc} on ticket {tkt}.",
            [("name", "Hiroshi Tanaka", "PERSON"),
             ("ph", "+81 3 1234 5678", "PHONE"),
             ("cc", "4012888888881881", "CREDIT_CARD"),
             ("tkt", "TKT-77201", "TICKET_ID")]
        ),
        (
            "Dear {name}, your password {pwd} for database {ip} will expire on {dt}.",
            [("name", "Nadia Petrova", "PERSON"),
             ("pwd", "Nadia#Pass8812", "PASSWORD"),
             ("ip", "10.50.2.19", "IP_ADDRESS"),
             ("dt", "30 November 2024", "DATE")]
        ),
        (
            "Support specialist {name} at {org} closed ticket {tkt} for customer {cid} on {dt}.",
            [("name", "Ananya Sen", "PERSON"),
             ("org", "Cognizant", "ORGANIZATION"),
             ("tkt", "TKT-33910", "TICKET_ID"),
             ("cid", "CUST-19402", "CUSTOMER_ID"),
             ("dt", "18/07/2024", "DATE")]
        ),
        (
            "Transfer 12,000 USD from account {acc} to card {cc} for customer {name} in {loc}.",
            [("acc", "998877661122", "BANK_ACCOUNT"),
             ("cc", "4222222222222", "CREDIT_CARD"),
             ("name", "Lucas Silva", "PERSON"),
             ("loc", "Sao Paulo", "LOCATION")]
        ),
        (
            "Please send the contract to {name} at {addr}. Email: {email}, Phone: {ph}.",
            [("name", "Claire Dupont", "PERSON"),
             ("addr", "45 Boulevard Saint-Germain, 75005 Paris", "ADDRESS"),
             ("email", "claire.dupont@avocat.fr", "EMAIL"),
             ("ph", "+33 1 42 68 00 00", "PHONE")]
        ),

        # --- 8. Public Knowledge (Famous Figures, Companies, Tech) ---
        (
            "{name} founded {org1} and also oversees {org2} based in {loc}.",
            [("name", "Elon Musk", "PERSON"),
             ("org1", "Tesla", "ORGANIZATION"),
             ("org2", "SpaceX", "ORGANIZATION"),
             ("loc", "Austin", "LOCATION")]
        ),
        (
            "{name} succeeded Steve Jobs as CEO of {org} in {loc}.",
            [("name", "Tim Cook", "PERSON"),
             ("org", "Apple", "ORGANIZATION"),
             ("loc", "Cupertino", "LOCATION")]
        ),
        (
            "{name} leads {org} and announced new AI chips in {loc}.",
            [("name", "Jensen Huang", "PERSON"),
             ("org", "Nvidia", "ORGANIZATION"),
             ("loc", "Santa Clara", "LOCATION")]
        ),
        (
            "{name} serves as the chief executive officer of {org} in {loc}.",
            [("name", "Satya Nadella", "PERSON"),
             ("org", "Microsoft", "ORGANIZATION"),
             ("loc", "Redmond", "LOCATION")]
        ),
        (
            "{name} co-founded {org} and co-created ChatGPT.",
            [("name", "Sam Altman", "PERSON"),
             ("org", "OpenAI", "ORGANIZATION")]
        ),
        (
            "{name} created Linux and Git while living in {loc}.",
            [("name", "Linus Torvalds", "PERSON"),
             ("loc", "Helsinki", "LOCATION")]
        ),

        # --- 9. Negative Controls (Zero-PII Prompts: Technical, Math, Academic) ---
        (
            "Binary search has a logarithmic time complexity of O(log n) in the worst case.",
            []
        ),
        (
            "The quick brown fox jumps over the lazy dog in standard English typography tests.",
            []
        ),
        (
            "Photosynthesis converts carbon dioxide and water into glucose and oxygen using sunlight.",
            []
        ),
        (
            "Database normalization reduces data redundancy and improves data integrity across tables.",
            []
        ),
        (
            "In mathematics, the Pythagorean theorem states that a squared plus b squared equals c squared.",
            []
        ),
        (
            "HTTP 404 is an error status code indicating that the server cannot find the requested resource.",
            []
        ),
        (
            "TCP provides reliable, ordered, and error-checked delivery of a stream of octets between applications.",
            []
        ),
        (
            "Machine learning models utilize gradient descent optimization to minimize the loss function.",
            []
        ),
        (
            "Recursion occurs when a function calls itself directly or indirectly to solve smaller subproblems.",
            []
        ),
        (
            "Unit testing verifies that individual components of software work as expected in isolation.",
            []
        ),
        (
            "Continuous integration pipelines automatically build, test, and package application artifacts.",
            []
        ),
        (
            "Refactoring improves internal code structure without altering external observable behavior.",
            []
        ),
        (
            "The capital asset pricing model estimates the expected return of an asset given its systematic risk.",
            []
        ),
        (
            "Object-oriented programming principles include encapsulation, abstraction, inheritance, and polymorphism.",
            []
        ),
        (
            "Hash tables offer average O(1) time complexity for lookup, insertion, and deletion operations.",
            []
        ),
    ]

    # Expand the specs into 145 unique prompts by generating variations
    prompts = []
    pid = start_id

    # Add the primary base specs first
    for template, entities in raw_specs:
        if pid > 250:
            break
        prompt_data = build_prompt_with_entities(template, entities)
        prompt_data["id"] = f"test_{pid:03d}"
        prompts.append(prompt_data)
        pid += 1

    # Systematic generative variations to reach exactly 250 prompts
    var_entities_pool = [
        # Set A: Personal profiles
        ("Priya Sundaram", "p.sundaram@chennaitech.co.in", "+91 9840112233", "Infosys", "Chennai"),
        ("Alexander Wright", "a.wright@manchester.ac.uk", "+44 161 275 0000", "Barclays", "Manchester"),
        ("Sora Yamamoto", "sora.y@tokyo-robotics.jp", "+81 3 5555 0199", "Sony", "Tokyo"),
        ("Beatriz Morales", "bmorales@madrid-telecom.es", "+34 91 123 4567", "Telefonica", "Madrid"),
        ("Ibrahim Hassan", "ibrahim.h@cairo-data.eg", "+20 2 2770 0000", "Vodafone", "Cairo"),
        ("Zoe Tremblay", "zoe.t@montreal-ai.ca", "+1 514 555 0188", "Element AI", "Montreal"),
        ("Johan Lindqvist", "j.lindqvist@stockholm-fin.se", "+46 8 123 4567", "Klarna", "Stockholm"),
        ("Mei-Ling Chen", "meiling.chen@taipei-chip.tw", "+886 2 2345 6789", "TSMC", "Taipei"),
        ("Dmitri Ivanov", "dmitri.ivanov@moscow-sec.ru", "+7 495 123-45-67", "Kaspersky", "Moscow"),
        ("Chioma Okonjo", "c.okonjo@lagos-pay.ng", "+234 1 234 5678", "Flutterwave", "Lagos"),
    ]

    idx = 0
    while pid <= 250:
        name, email, ph, org, loc = var_entities_pool[idx % len(var_entities_pool)]
        t_choice = (pid - start_id) % 8

        if t_choice == 0:
            template = "Contact lead engineer {name} from {org} at {email} regarding project deployment in {loc}."
            entities = [("name", name, "PERSON"), ("org", org, "ORGANIZATION"), ("email", email, "EMAIL"), ("loc", loc, "LOCATION")]
        elif t_choice == 1:
            template = "Customer {name} reached support by phone at {ph} concerning invoice order {ord}."
            entities = [("name", name, "PERSON"), ("ph", ph, "PHONE"), ("ord", f"ORD-{70000 + pid}", "ORDER_ID")]
        elif t_choice == 2:
            template = "System administrator {name} renewed the SSL token for internal server {ip} on {dt}."
            entities = [("name", name, "PERSON"), ("ip", f"10.{pid % 50}.4.{pid % 250}", "IP_ADDRESS"), ("dt", f"{(pid % 28) + 1:02d}/11/2024", "DATE")]
        elif t_choice == 3:
            template = "Please update account credentials for client {name} (ID: {cid}). Temporary password is {pwd}."
            entities = [("name", name, "PERSON"), ("cid", f"CUST-{80000 + pid}", "CUSTOMER_ID"), ("pwd", f"PassKey#{pid * 17}!", "PASSWORD")]
        elif t_choice == 4:
            template = "Support incident {tkt} filed by {email} was resolved by technician {name} in {loc}."
            entities = [("tkt", f"TKT-{50000 + pid}", "TICKET_ID"), ("email", email, "EMAIL"), ("name", name, "PERSON"), ("loc", loc, "LOCATION")]
        elif t_choice == 5:
            template = "Wire the consulting payment to account {acc} under {name} at {org}."
            entities = [("acc", f"{900000000000 + pid * 13}", "BANK_ACCOUNT"), ("name", name, "PERSON"), ("org", org, "ORGANIZATION")]
        elif t_choice == 6:
            template = "Algorithm efficiency can be benchmarked using asymptotic notation and empirical runtime profiling."
            entities = []  # Zero-PII control
        else:
            template = "Encrypted communication channels rely on public key cryptography to establish secure shared secrets."
            entities = []  # Zero-PII control

        prompt_data = build_prompt_with_entities(template, entities)
        prompt_data["id"] = f"test_{pid:03d}"
        prompts.append(prompt_data)
        pid += 1
        idx += 1

    return prompts


def main():
    print(f"Loading existing ground truth from: {GT_FILE}")
    with open(GT_FILE, "r", encoding="utf-8") as f:
        existing_data = json.load(f)

    existing_count = len(existing_data)
    print(f"Existing prompts count: {existing_count}")

    if existing_count >= 250:
        print("Dataset already contains 250 or more prompts.")
        return

    needed = 250 - existing_count
    start_id = existing_count + 1
    print(f"Generating {needed} new prompts starting from test_{start_id:03d} to test_250...")

    new_prompts = generate_145_new_prompts(start_id=start_id)
    combined = existing_data + new_prompts

    print(f"Total prompts after merge: {len(combined)}")
    assert len(combined) == 250, f"Expected 250 prompts, got {len(combined)}"

    # Validate character offset integrity across ALL 250 prompts
    print("\nValidating 100% character offset integrity across all 250 prompts...")
    total_entities = 0
    errors = []
    for item in combined:
        text = item["text"]
        for e in item.get("entities", []):
            total_entities += 1
            actual = text[e["start"]:e["end"]]
            if actual != e["text"]:
                errors.append(f"[{item['id']}] Mismatch: expected '{e['text']}', got '{actual}' in text: {text}")

    if errors:
        print(f"[FATAL] Found {len(errors)} offset errors:")
        for err in errors[:10]:
            print("  ", err)
        sys.exit(1)

    print(f"[SUCCESS] All {total_entities} entity offsets verified with 100% exact slice matching!")

    # Write backup and updated file
    backup_file = BENCHMARK_DIR / f"ground_truth_backup_105.json"
    with open(backup_file, "w", encoding="utf-8") as f:
        json.dump(existing_data, f, indent=2, ensure_ascii=False)
    print(f"Original 105 prompts backed up to: {backup_file}")

    with open(GT_FILE, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)
    print(f"Successfully saved 250 prompts to: {GT_FILE}")


if __name__ == "__main__":
    main()
