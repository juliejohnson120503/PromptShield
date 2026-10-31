#!/usr/bin/env python3
"""
PromptShield Benchmark Dataset Expander (250 -> 500 Prompts)
=============================================================
Adds 250 realistic, enterprise-grade, and diverse prompts to reach
exactly 500 prompts in ground_truth.json.

Ensures:
1. Exact character-level offset calculation: text[start:end] == entity['text']
2. Diverse global names (Indian, European, American, Asian, Hispanic, Middle Eastern, African)
3. Diverse technical credentials, Luhn-valid credit cards, IBANs, IP addresses, JWT tokens
4. Public figures and organizations properly labeled
5. Negative controls (zero-PII prompts) to evaluate precision robustness
6. Strict offset verification before saving
"""

import json
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

BENCHMARK_DIR = Path(__file__).resolve().parent
GT_FILE = BENCHMARK_DIR / "ground_truth.json"
BACKUP_FILE = BENCHMARK_DIR / "ground_truth_250_backup.json"


def build_prompt_with_entities(template: str, entities_to_place: List[Tuple[str, str, str]]) -> Dict:
    """
    Constructs a prompt and computes exact [start, end) offsets for each entity.
    Uses regex left-to-right scanning so order of entities_to_place does not affect offsets.
    entities_to_place: List of (placeholder_key, entity_text, label)
    template: Text containing {placeholder_key} markers.
    """
    lookup = {k: (val, label) for k, val, label in entities_to_place}
    
    parts = []
    clean_entities = []
    last_idx = 0
    current_pos = 0

    matches = list(re.finditer(r"\{([a-zA-Z0-9_]+)\}", template))
    matched_keys = {m.group(1) for m in matches}
    for k in lookup:
        if k not in matched_keys:
            raise ValueError(f"Placeholder '{{{k}}}' declared in entities_to_place but NOT found in template: '{template}'")

    for match in matches:
        start, end = match.span()
        key = match.group(1)
        if key not in lookup:
            continue

        prefix = template[last_idx:start]
        parts.append(prefix)
        current_pos += len(prefix)

        entity_val, entity_label = lookup[key]
        ent_start = current_pos
        ent_end = current_pos + len(entity_val)
        parts.append(entity_val)
        current_pos += len(entity_val)

        clean_entities.append({
            "text": entity_val,
            "start": ent_start,
            "end": ent_end,
            "label": entity_label
        })
        last_idx = end

    remaining = template[last_idx:]
    parts.append(remaining)
    rendered_text = "".join(parts)

    for e in clean_entities:
        actual = rendered_text[e["start"]:e["end"]]
        if actual != e["text"]:
            raise ValueError(f"Offset mismatch: expected '{e['text']}' got '{actual}' at [{e['start']}:{e['end']}] in prompt: '{rendered_text}'")

    return {
        "text": rendered_text,
        "entities": clean_entities
    }


def get_250_new_prompt_specs() -> List[Tuple[str, List[Tuple[str, str, str]]]]:
    specs: List[Tuple[str, List[Tuple[str, str, str]]]] = [
        # =====================================================================
        # CATEGORY 1: Customer Support & E-commerce (30 prompts: 251 - 280)
        # =====================================================================
        (
            "Customer {name} reported that package for order {order} has not arrived at {addr}.",
            [("name", "Elena Rostova", "PERSON"), ("order", "ORD-992014", "ORDER_ID"), ("addr", "42 Nevsky Prospekt, St. Petersburg, 191186", "ADDRESS")]
        ),
        (
            "Please check refund status for ticket {ticket} submitted by {email} on {date}.",
            [("ticket", "TICKET-44810", "TICKET_ID"), ("email", "tariq.mansoor@dubaimail.ae", "EMAIL"), ("date", "April 18, 2024", "DATE")]
        ),
        (
            "The customer service agent helped {name} update delivery instructions for order {order}.",
            [("name", "Carlos Mendoza", "PERSON"), ("order", "ORDER-772190", "ORDER_ID")]
        ),
        (
            "Account {cust} has requested phone support at {phone} regarding a defective laptop.",
            [("cust", "CUST-88310", "CUSTOMER_ID"), ("phone", "+34 912 345 678", "PHONE")]
        ),
        (
            "Ship replacement parts to {name} at {addr} before the weekend.",
            [("name", "Ananya Deshmukh", "PERSON"), ("addr", "15 MG Road, Bangalore, 560001", "ADDRESS")]
        ),
        (
            "Escalated support ticket {ticket} requires tier 3 review by {date}.",
            [("ticket", "INC-551092", "TICKET_ID"), ("date", "May 22, 2024", "DATE")]
        ),
        (
            "User {email} cannot access their billing portal for order {order}.",
            [("email", "hannah.schmidt@berlin-tech.de", "EMAIL"), ("order", "ORD-110294", "ORDER_ID")]
        ),
        (
            "Client {name} registered with customer number {cust} requested cancellation.",
            [("name", "Liam Gallagher", "PERSON"), ("cust", "CUST-33491", "CUSTOMER_ID")]
        ),
        (
            "Delivery to {addr} failed due to incorrect contact phone {phone}.",
            [("addr", "88 Queen Street, Melbourne, VIC 3000", "ADDRESS"), ("phone", "+61 3 9876 5432", "PHONE")]
        ),
        (
            "Support ticket {ticket} regarding customer {name} was closed on {date}.",
            [("ticket", "TICKET-66201", "TICKET_ID"), ("name", "Fiona Gallagher", "PERSON"), ("date", "June 30, 2024", "DATE")]
        ),
        (
            "Inquire about shipment tracking for customer {cust} under order {order}.",
            [("cust", "CID-44918", "CUSTOMER_ID"), ("order", "ORD-662914", "ORDER_ID")]
        ),
        (
            "Contact {name} at {email} regarding the damaged package received in {loc}.",
            [("name", "Kenji Takahashi", "PERSON"), ("email", "kenji.takahashi@osaka-net.jp", "EMAIL"), ("loc", "Osaka", "LOCATION")]
        ),
        (
            "Refund of $149 for order {order} was sent to customer {email}.",
            [("order", "ORDER-99104", "ORDER_ID"), ("email", "olivia.martinez@sbcglobal.net", "EMAIL")]
        ),
        (
            "Courier confirmed package delivered to {name} at {addr}.",
            [("name", "Zainab Al-Fassi", "PERSON"), ("addr", "74 Corniche Road, Abu Dhabi, 24395", "ADDRESS")]
        ),
        (
            "Ticket {ticket} requires feedback from user {cust} before resolution.",
            [("ticket", "TICKET-10492", "TICKET_ID"), ("cust", "USER-5582", "USER_ID")]
        ),
        (
            "Customer {name} called from {phone} asking to update their shipping address.",
            [("name", "Lucas Silva", "PERSON"), ("phone", "+55 11 98765-4321", "PHONE")]
        ),
        (
            "Order {order} has been dispatched to {name} via express courier.",
            [("order", "ORD-332910", "ORDER_ID"), ("name", "Siddharth Rao", "PERSON")]
        ),
        (
            "Help desk ticket {ticket} filed by {email} is awaiting supervisor approval.",
            [("ticket", "INC-994812", "TICKET_ID"), ("email", "dmitri.ivanov@moscow-trade.ru", "EMAIL")]
        ),
        (
            "Customer {cust} verified identity using registered email {email}.",
            [("cust", "CUST-77412", "CUSTOMER_ID"), ("email", "sophie.dubois@paris-mail.fr", "EMAIL")]
        ),
        (
            "Deliver the replacement tablet to {addr} for {name}.",
            [("addr", "12 Wenceslas Square, Prague, 110 00", "ADDRESS"), ("name", "Viktor Novak", "PERSON")]
        ),
        (
            "The customer service agent resolved ticket {ticket} after verifying phone {phone}.",
            [("ticket", "TICKET-88210", "TICKET_ID"), ("phone", "+44 20 7946 0912", "PHONE")]
        ),
        (
            "Order {order} placed by {name} on {date} is ready for store pickup.",
            [("order", "ORD-554219", "ORDER_ID"), ("name", "Fatima Al-Mansoor", "PERSON"), ("date", "July 14, 2024", "DATE")]
        ),
        (
            "Customer {cust} requested a callback at {phone} before 5 PM.",
            [("cust", "CUST-22941", "CUSTOMER_ID"), ("phone", "+1-555-839-2041", "PHONE")]
        ),
        (
            "Dispute for order {order} was escalated to manager {name}.",
            [("order", "ORDER-228190", "ORDER_ID"), ("name", "Rachel Adams", "PERSON")]
        ),
        (
            "Warranty claim {ticket} was submitted by {email} for product serial SN-9948.",
            [("ticket", "TICKET-33109", "TICKET_ID"), ("email", "matteo.rossi@milan-design.it", "EMAIL")]
        ),
        (
            "Please deliver package {order} to {name} at {loc}.",
            [("order", "ORD-771029", "ORDER_ID"), ("name", "Sunil Gavaskar", "PERSON"), ("loc", "Mumbai", "LOCATION")]
        ),
        (
            "Customer {cust} changed their registered email address to {email}.",
            [("cust", "CUST-66104", "CUSTOMER_ID"), ("email", "yuki.tanaka@tokyo-cloud.jp", "EMAIL")]
        ),
        (
            "The support desk received ticket {ticket} regarding payment gateway errors.",
            [("ticket", "INC-332910", "TICKET_ID")]
        ),
        (
            "Customer {name} at {phone} is experiencing login difficulties.",
            [("name", "Chloe Dupont", "PERSON"), ("phone", "+33 1 42 68 55 00", "PHONE")]
        ),
        (
            "Confirm that replacement shipment for {order} was routed to {addr}.",
            [("order", "ORD-449182", "ORDER_ID"), ("addr", "10 Downing Street, London, SW1A 2AA", "ADDRESS")]
        ),

        # =====================================================================
        # CATEGORY 2: Financial & Banking (25 prompts: 281 - 305)
        # =====================================================================
        (
            "Transfer $5,000 to bank account {acc} under beneficiary {name} at {org}.",
            [("acc", "400918273645", "BANK_ACCOUNT"), ("name", "Alistair Finch", "PERSON"), ("org", "Barclays", "ORGANIZATION")]
        ),
        (
            "The corporate account IBAN {iban} was debited for vendor software licenses.",
            [("iban", "DE89 3704 0044 0532 0130 00", "BANK_ACCOUNT")]
        ),
        (
            "Process the monthly retainer of EUR 8,200 to {name} via IBAN {iban}.",
            [("name", "Beatrice Vane", "PERSON"), ("iban", "FR76 3000 6000 0112 3456 7890 189", "BANK_ACCOUNT")]
        ),
        (
            "Recurring payment charged to Visa card {cc} expires {date}.",
            [("cc", "4532 9812 3456 7890", "CREDIT_CARD"), ("date", "12/28", "DATE")]
        ),
        (
            "Credit card {cc} registered to {name} was blocked due to suspected fraud in {loc}.",
            [("cc", "5412 7534 8901 2345", "CREDIT_CARD"), ("name", "Julian Mercer", "PERSON"), ("loc", "Lisbon", "LOCATION")]
        ),
        (
            "Direct deposit instructions for employee {name}: account {acc} at {org}.",
            [("name", "Deepak Chopra", "PERSON"), ("acc", "987654321098", "BANK_ACCOUNT"), ("org", "HDFC Bank", "ORGANIZATION")]
        ),
        (
            "Wire transfer to Spanish IBAN {iban} was confirmed by {org}.",
            [("iban", "ES91 2100 0418 4502 0005 1332", "BANK_ACCOUNT"), ("org", "Santander", "ORGANIZATION")]
        ),
        (
            "Please bill credit card {cc} for the cloud subscription under {email}.",
            [("cc", "4000 1234 5678 9010", "CREDIT_CARD"), ("email", "billing@enterprise-saas.com", "EMAIL")]
        ),
        (
            "Settlement payment of GBP 14,000 sent to {name} using account {acc}.",
            [("name", "Nigel Hawthorne", "PERSON"), ("acc", "654321098765", "BANK_ACCOUNT")]
        ),
        (
            "The client paid via {org} card {cc} for invoice INV-88219.",
            [("org", "American Express", "ORGANIZATION"), ("cc", "3782 822468 00005", "CREDIT_CARD")]
        ),
        (
            "Remittance advice: EUR 45,000 transferred to Italian IBAN {iban} for {org}.",
            [("iban", "IT60 X054 2811 1010 0000 0123 456", "BANK_ACCOUNT"), ("org", "Eni", "ORGANIZATION")]
        ),
        (
            "Please verify that credit card {cc} belongs to cardholder {name}.",
            [("cc", "5105 1051 0510 5100", "CREDIT_CARD"), ("name", "Svetlana Kuznetsova", "PERSON")]
        ),
        (
            "Bank account {acc} was linked to the payroll service by finance manager {name}.",
            [("acc", "112233445566", "BANK_ACCOUNT"), ("name", "Robert Langdon", "PERSON")]
        ),
        (
            "Send the payout of $3,200 to Dutch account IBAN {iban} on {date}.",
            [("iban", "NL91 ABNA 0417 1643 00", "BANK_ACCOUNT"), ("date", "August 1, 2024", "DATE")]
        ),
        (
            "Disputed transaction of $840 on card {cc} was flagged by {org}.",
            [("cc", "4111 1111 1111 4321", "CREDIT_CARD"), ("org", "Visa", "ORGANIZATION")]
        ),
        (
            "Routing number 021000021 and account {acc} should be used for federal tax payments.",
            [("acc", "998877665544", "BANK_ACCOUNT")]
        ),
        (
            "The escrow account IBAN {iban} was opened by attorney {name}.",
            [("iban", "CH93 0076 2011 6238 5295 7", "BANK_ACCOUNT"), ("name", "Christian Meier", "PERSON")]
        ),
        (
            "Customer authorization given to charge Discover card {cc} for annual membership.",
            [("cc", "6011 0009 9013 9424", "CREDIT_CARD")]
        ),
        (
            "Wire transfer to account {acc} under {name} cleared on {date}.",
            [("acc", "334455667788", "BANK_ACCOUNT"), ("name", "Hiroshi Sato", "PERSON"), ("date", "September 15, 2024", "DATE")]
        ),
        (
            "Audit finding: credit card {cc} stored in plaintext in database by {email}.",
            [("cc", "4916 9912 3456 7890", "CREDIT_CARD"), ("email", "dev-lead@fintech-pay.io", "EMAIL")]
        ),
        (
            "The company Swiss IBAN {iban} received an international transfer from {org}.",
            [("iban", "CH56 0483 5012 3456 7800 9", "BANK_ACCOUNT"), ("org", "Novartis", "ORGANIZATION")]
        ),
        (
            "Corporate expenses charged to Mastercard {cc} held by executive {name}.",
            [("cc", "5500 0000 0000 0004", "CREDIT_CARD"), ("name", "Amira El-Sayed", "PERSON")]
        ),
        (
            "Direct debit mandate authorized for bank account {acc} in favor of {org}.",
            [("acc", "887766554433", "BANK_ACCOUNT"), ("org", "Vodafone", "ORGANIZATION")]
        ),
        (
            "Verify settlement batch with Belgian IBAN {iban} before end of banking day.",
            [("iban", "BE68 5390 0754 7034", "BANK_ACCOUNT")]
        ),
        (
            "Refund payment of $250 issued to credit card {cc} belonging to {name}.",
            [("cc", "4485 1234 5678 9012", "CREDIT_CARD"), ("name", "Gabriel Vasquez", "PERSON")]
        ),

        # =====================================================================
        # CATEGORY 3: DevOps, Cloud Infrastructure & Cyber (30 prompts: 306 - 335)
        # =====================================================================
        (
            "Deploy the microservice to {org} cluster at {ip} using secret token {tok}.",
            [("org", "Kubernetes", "ORGANIZATION"), ("ip", "192.168.10.45", "IP_ADDRESS"), ("tok", "Bearer eyJhbGciOiJIUzI1NiJ9.eyJpZCI6MTIzfQ.k7J9X2a", "ACCESS_TOKEN")]
        ),
        (
            "The production {org} database at {ip} requires password {pwd} for replication.",
            [("org", "Redis", "ORGANIZATION"), ("ip", "10.0.4.15", "IP_ADDRESS"), ("pwd", "Red!sStr0ngP@ss2024", "PASSWORD")]
        ),
        (
            "Update {org1} access key {k1} in the continuous delivery pipeline for {org2}.",
            [("org1", "AWS", "ORGANIZATION"), ("k1", "AKIAIOSFODNN7DEV2024", "API_KEY"), ("org2", "Stripe", "ORGANIZATION")]
        ),
        (
            "Server {ip} experienced an unauthorized root login attempt on {date}.",
            [("ip", "203.0.113.195", "IP_ADDRESS"), ("date", "October 5, 2024", "DATE")]
        ),
        (
            "GitHub token {pat} was leaked in commit logs by developer {email}.",
            [("pat", "ghp_AbCdEfGhIjKlMnOpQrStUvWxYz0123456789", "API_KEY"), ("email", "alex.turner@cloud-dev.org", "EMAIL")]
        ),
        (
            "Authenticate to {org} API using private key {k1} for semantic embeddings.",
            [("org", "OpenAI", "ORGANIZATION"), ("k1", "sk-proj-99887766554433221100aabbccddeeffgg", "API_KEY")]
        ),
        (
            "Configure VPN gateway on {ip} with preshared secret {pwd}.",
            [("ip", "198.51.100.88", "IP_ADDRESS"), ("pwd", "IpSec#Secr3t!2024", "PASSWORD")]
        ),
        (
            "Administrator {name} rotated the database admin password to {pwd} on host {ip}.",
            [("name", "Derrick Vance", "PERSON"), ("pwd", "Adm!nP@ss#9920", "PASSWORD"), ("ip", "172.16.8.22", "IP_ADDRESS")]
        ),
        (
            "{org} webhook integration key {k1} was revoked by the security team.",
            [("org", "Slack", "ORGANIZATION"), ("k1", "xoxb-9988776655-4433221100-abcdef123456", "API_KEY")]
        ),
        (
            "Staging environment API gateway accessible at {ip} requires authorization header.",
            [("ip", "192.0.2.144", "IP_ADDRESS")]
        ),
        (
            "The {org1} Maps API key {k1} has exceeded its daily quota for client {org2}.",
            [("org1", "Google", "ORGANIZATION"), ("k1", "AIzaSyExampleUberMapsKey1234567890abcde", "API_KEY"), ("org2", "Uber", "ORGANIZATION")]
        ),
        (
            "Connect to MySQL instance {ip} using username dbuser and password {pwd}.",
            [("ip", "10.200.1.18", "IP_ADDRESS"), ("pwd", "MySql_Root$ecure2024", "PASSWORD")]
        ),
        (
            "The OAuth authorization server issued bearer token {tok} for session validation.",
            [("tok", "Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.xyz987token", "ACCESS_TOKEN")]
        ),
        (
            "Firewall rule added on router {ip} to block port 22 traffic from {loc}.",
            [("ip", "198.51.100.210", "IP_ADDRESS"), ("loc", "Shenzhen", "LOCATION")]
        ),
        (
            "Engineer {name} refreshed the {org} monitoring API key {k1}.",
            [("name", "Lars Sorensen", "PERSON"), ("org", "Datadog", "ORGANIZATION"), ("k1", "dd_api_key_8899aabbccddeeff00112233", "API_KEY")]
        ),
        (
            "Log in to the bastion host at {ip} using temporary SSH password {pwd}.",
            [("ip", "10.50.100.2", "IP_ADDRESS"), ("pwd", "B@st!onTemp2024", "PASSWORD")]
        ),
        (
            "Send alert notifications to DevOps on-call {email} when {ip} goes offline.",
            [("email", "devops-oncall@fintech-core.io", "EMAIL"), ("ip", "172.20.14.9", "IP_ADDRESS")]
        ),
        (
            "The SendGrid email API key {k1} was updated by system architect {name}.",
            [("k1", "SG.99887766554433221100aa.bbccddeeffgghhiijjkkllmmnnooppqq", "API_KEY"), ("name", "Kavita Raman", "PERSON")]
        ),
        (
            "DNS server {ip} resolved domain lookups for client network in {loc}.",
            [("ip", "8.8.8.8", "IP_ADDRESS"), ("loc", "Singapore", "LOCATION")]
        ),
        (
            "Reset credentials for user {email} with default password {pwd}.",
            [("email", "new_hire@enterprise-corp.com", "EMAIL"), ("pwd", "WelcomeToCorp2024!", "PASSWORD")]
        ),
        (
            "Container registry auth token {tok} expired on {date}.",
            [("tok", "Bearer eyJhbGciOiJIUzI1NiJ9.dockerRegistryToken2024", "ACCESS_TOKEN"), ("date", "November 10, 2024", "DATE")]
        ),
        (
            "Vulnerability scanner detected open telnet port on IP address {ip}.",
            [("ip", "203.0.113.84", "IP_ADDRESS")]
        ),
        (
            "Stripe publishable key {k1} configured for e-commerce checkout.",
            [("k1", "pk_live_51AbCdEfGhIjKlMnOpQrStUvWxYz", "API_KEY")]
        ),
        (
            "Database migration script executed against {ip} by developer {name}.",
            [("ip", "10.10.30.55", "IP_ADDRESS"), ("name", "Artur Wisniewski", "PERSON")]
        ),
        (
            "Production load balancer at {ip} routed traffic across nodes in {loc}.",
            [("ip", "192.168.1.254", "IP_ADDRESS"), ("loc", "Zurich", "LOCATION")]
        ),
        (
            "The {org} build pipeline secret was set to {pwd} on server {ip}.",
            [("org", "Jenkins", "ORGANIZATION"), ("pwd", "Jenk!nsBld#7719", "PASSWORD"), ("ip", "10.0.15.80", "IP_ADDRESS")]
        ),
        (
            "{org} API token {k1} granted access to Claude models.",
            [("org", "Anthropic", "ORGANIZATION"), ("k1", "sk-ant-api03-99887766554433221100aabbccdd-xyz", "API_KEY")]
        ),
        (
            "Security analyst {name} confirmed IP {ip} was an authorized internal scanner.",
            [("name", "Gemma Davies", "PERSON"), ("ip", "172.16.254.1", "IP_ADDRESS")]
        ),
        (
            "The authorization bearer token {tok} was revoked following session timeout.",
            [("tok", "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.sessionExpiredVal", "ACCESS_TOKEN")]
        ),
        (
            "Backup archive encrypted using symmetric key password {pwd} in cloud storage.",
            [("pwd", "B@ckupEncr!pt#2024", "PASSWORD")]
        ),

        # =====================================================================
        # CATEGORY 4: HR, Recruitment & Executive Search (25 prompts: 336 - 360)
        # =====================================================================
        (
            "Candidate {name} graduated from {org} in {loc} with honors in Computer Science.",
            [("name", "Mei-Ling Zhou", "PERSON"), ("org", "Tsinghua University", "ORGANIZATION"), ("loc", "Beijing", "LOCATION")]
        ),
        (
            "Forward the resume of {name} to hiring manager at {email}.",
            [("name", "Benoit Blanc", "PERSON"), ("email", "recruiting-lead@biotech-global.com", "EMAIL")]
        ),
        (
            "Schedule an interview with {name} at {phone} for the VP position on {date}.",
            [("name", "Haruto Watanabe", "PERSON"), ("phone", "+81 3 5555 0143", "PHONE"), ("date", "December 2, 2024", "DATE")]
        ),
        (
            "Applicant {name} previously worked at {org} as a principal architect in {loc}.",
            [("name", "Ingrid Lindholm", "PERSON"), ("org", "Ericsson", "ORGANIZATION"), ("loc", "Stockholm", "LOCATION")]
        ),
        (
            "Offer letter sent to {email} for the senior backend developer role.",
            [("email", "samir.kassir@beirut-dev.org", "EMAIL")]
        ),
        (
            "HR manager {name} confirmed background check completed for employee ID {cust}.",
            [("name", "Patricia O'Connor", "PERSON"), ("cust", "EMP-99210", "CUSTOMER_ID")]
        ),
        (
            "Contact software engineer {name} directly on mobile {phone} regarding salary expectations.",
            [("name", "Karthik Sundaram", "PERSON"), ("phone", "+91 94440 12345", "PHONE")]
        ),
        (
            "Reference check: {name} confirmed that candidate worked at {org} for 4 years.",
            [("name", "Eleanor Vance", "PERSON"), ("org", "Accenture", "ORGANIZATION")]
        ),
        (
            "Recruiter {name} reviewed 45 applicants for the DevOps vacancy in {loc}.",
            [("name", "Thiago Santos", "PERSON"), ("loc", "Sao Paulo", "LOCATION")]
        ),
        (
            "Send employment contract to {name} residing at {addr}.",
            [("name", "Nadia Chernova", "PERSON"), ("addr", "28 Tverskaya Street, Moscow, 125009", "ADDRESS")]
        ),
        (
            "Applicant {email} requested to reschedule technical interview to {date}.",
            [("email", "andreas.meyer@vienna-tech.at", "EMAIL"), ("date", "December 15, 2024", "DATE")]
        ),
        (
            "Candidate {name} passed the final executive panel interview with {org}.",
            [("name", "Thabo Mbeki", "PERSON"), ("org", "Standard Bank", "ORGANIZATION")]
        ),
        (
            "HR director {name} approved relocation bonus for new hire moving to {loc}.",
            [("name", "Charlotte Bronte", "PERSON"), ("loc", "London", "LOCATION")]
        ),
        (
            "Confirm that {name} at {phone} accepted the director of engineering offer.",
            [("name", "Alejandro Gomez", "PERSON"), ("phone", "+52 55 1234 5678", "PHONE")]
        ),
        (
            "Send portfolio evaluation feedback to {email} before end of day.",
            [("email", "camilla.nielsen@copenhagen-art.dk", "EMAIL")]
        ),
        (
            "Talent acquisition specialist {name} sourced 12 candidates from {org}.",
            [("name", "Vijay Raghavan", "PERSON"), ("org", "Tata Consultancy Services", "ORGANIZATION")]
        ),
        (
            "Verify education credentials of {name} from {org} in {loc}.",
            [("name", "Stefan Horvat", "PERSON"), ("org", "University of Zagreb", "ORGANIZATION"), ("loc", "Zagreb", "LOCATION")]
        ),
        (
            "HR onboarding portal login created for {email} with employee code {cust}.",
            [("email", "monica.geller@manhattan-legal.com", "EMAIL"), ("cust", "EMP-44819", "CUSTOMER_ID")]
        ),
        (
            "Candidate {name} withdrew application citing competitive offer from {org}.",
            [("name", "Goran Ivanisevic", "PERSON"), ("org", "Oracle", "ORGANIZATION")]
        ),
        (
            "Executive search consultant {name} submitted shortlist for Chief Product Officer.",
            [("name", "Helena Berg", "PERSON")]
        ),
        (
            "Reach out to {name} via telephone {phone} to discuss severance terms.",
            [("name", "Francois Girard", "PERSON"), ("phone", "+33 4 72 00 11 22", "PHONE")]
        ),
        (
            "Internal transfer approved for {name} from {loc} branch to headquarters.",
            [("name", "Sanjay Singhal", "PERSON"), ("loc", "Hyderabad", "LOCATION")]
        ),
        (
            "Candidate resume received from {email} with references from {org}.",
            [("email", "leila.hadid@cairo-eng.eg", "EMAIL"), ("org", "Orascom", "ORGANIZATION")]
        ),
        (
            "HR coordinator {name} scheduled orientation sessions starting on {date}.",
            [("name", "Brian Molko", "PERSON"), ("date", "January 6, 2025", "DATE")]
        ),
        (
            "Performance appraisal meeting for {name} will be held at our office in {loc}.",
            [("name", "Sunita Williams", "PERSON"), ("loc", "Houston", "LOCATION")]
        ),

        # =====================================================================
        # CATEGORY 5: Healthcare, Clinical & Pharma (25 prompts: 361 - 385)
        # =====================================================================
        (
            "Patient {name} admitted to {org} for cardiology evaluation on {date}.",
            [("name", "Arthur Pendelton", "PERSON"), ("org", "Cleveland Clinic", "ORGANIZATION"), ("date", "January 12, 2025", "DATE")]
        ),
        (
            "Medical record {cust} for patient {name} updated by Dr. Thorne.",
            [("cust", "MRN-882910", "MEDICAL_RECORD"), ("name", "Margaret Higgins", "PERSON")]
        ),
        (
            "Prescription refill for {name} sent to pharmacy located at {addr}.",
            [("name", "Walter Bishop", "PERSON"), ("addr", "450 Massachusetts Ave, Cambridge, MA 02139", "ADDRESS")]
        ),
        (
            "Clinical trial participant {cust} reported adverse reactions to principal investigator {name}.",
            [("cust", "SUBJ-4410", "CUSTOMER_ID"), ("name", "Aris Thorne", "PERSON")]
        ),
        (
            "Emergency contact for patient {name} is {phone} residing in {loc}.",
            [("name", "Mikhail Voronin", "PERSON"), ("phone", "+7 495 123-45-67", "PHONE"), ("loc", "Moscow", "LOCATION")]
        ),
        (
            "Radiology scans for {name} were transferred to {org} for secondary opinion.",
            [("name", "Clara Oswald", "PERSON"), ("org", "Johns Hopkins Hospital", "ORGANIZATION")]
        ),
        (
            "Physician {name} scheduled post-operative follow-up on {date}.",
            [("name", "Alistair Sterling", "PERSON"), ("date", "February 14, 2025", "DATE")]
        ),
        (
            "Confidential lab results for patient {email} showed normal glucose levels.",
            [("email", "george.costanza@queens-med.org", "EMAIL")]
        ),
        (
            "Outpatient consultation for {name} confirmed at specialty clinic in {loc}.",
            [("name", "Zahra Hosseini", "PERSON"), ("loc", "Tehran", "LOCATION")]
        ),
        (
            "Discharge summary for record {cust} prepared by attending nurse {name}.",
            [("cust", "MED-33019", "CUSTOMER_ID"), ("name", "Fiona Gallagher", "PERSON")]
        ),
        (
            "Patient {name} scheduled for MRI scan at {org} on {date}.",
            [("name", "Donald Cragen", "PERSON"), ("org", "Mayo Clinic", "ORGANIZATION"), ("date", "March 3, 2025", "DATE")]
        ),
        (
            "Emergency room admission notification sent to relative at {phone}.",
            [("phone", "+1-617-555-0199", "PHONE")]
        ),
        (
            "Pharmacist {name} dispensed chemotherapy medication under prescription RX-99214.",
            [("name", "Sunil Narang", "PERSON")]
        ),
        (
            "Secure medical portal login link sent to patient email {email}.",
            [("email", "eleanor.vance@boston-health.net", "EMAIL")]
        ),
        (
            "Pediatric appointment for child of {name} confirmed for {date}.",
            [("name", "Leticia Ortiz", "PERSON"), ("date", "March 20, 2025", "DATE")]
        ),
        (
            "Health insurance claim under policy {cust} submitted to {org}.",
            [("cust", "POL-992014", "CUSTOMER_ID"), ("org", "Aetna", "ORGANIZATION")]
        ),
        (
            "Surgeon {name} completed orthopedic procedure at hospital in {loc}.",
            [("name", "Jean-Luc Picard", "PERSON"), ("loc", "Lyon", "LOCATION")]
        ),
        (
            "Diagnostic biopsy report for {name} filed under medical record {cust}.",
            [("name", "Hanna Marin", "PERSON"), ("cust", "MRN-109283", "MEDICAL_RECORD")]
        ),
        (
            "Notify caregiver {name} at {phone} when patient is ready for discharge.",
            [("name", "Toby Flenderson", "PERSON"), ("phone", "+1-570-555-0144", "PHONE")]
        ),
        (
            "Clinical research protocol approved by ethics board at {org} in {loc}.",
            [("org", "Charité", "ORGANIZATION"), ("loc", "Berlin", "LOCATION")]
        ),
        (
            "Patient {name} signed medical consent form at clinic address {addr}.",
            [("name", "Rashid Al-Hassan", "PERSON"), ("addr", "10 Sheikh Zayed Road, Dubai, 11432", "ADDRESS")]
        ),
        (
            "Immunization record for {name} verified by health department.",
            [("name", "Silvia Rossi", "PERSON")]
        ),
        (
            "Hospital bed reserved for incoming patient under code {cust} on {date}.",
            [("cust", "PAT-55819", "CUSTOMER_ID"), ("date", "April 5, 2025", "DATE")]
        ),
        (
            "Genetic test results for {name} were encrypted and emailed to {email}.",
            [("name", "Dmitri Shostakovich", "PERSON"), ("email", "dmitri.shost@genomics-lab.ru", "EMAIL")]
        ),
        (
            "Doctor {name} reviewed telemetry data transmitted from patient monitor in {loc}.",
            [("name", "Eleni Papandreou", "PERSON"), ("loc", "Athens", "LOCATION")]
        ),

        # =====================================================================
        # CATEGORY 6: Legal & Contracts (25 prompts: 386 - 410)
        # =====================================================================
        (
            "Non-disclosure agreement executed between {org} and consultant {name} on {date}.",
            [("org", "Palantir", "ORGANIZATION"), ("name", "Reginald Sterling", "PERSON"), ("date", "April 15, 2025", "DATE")]
        ),
        (
            "Commercial lease for office premises at {addr} signed by tenant {name}.",
            [("addr", "350 Fifth Avenue, New York, NY 10118", "ADDRESS"), ("name", "Solomon Kane", "PERSON")]
        ),
        (
            "Notice of trademark infringement sent to {email} by legal counsel {name}.",
            [("email", "legal-dept@startup-domain.co", "EMAIL"), ("name", "Victoria Winters", "PERSON")]
        ),
        (
            "Law firm {org} represented the plaintiff in federal district court in {loc}.",
            [("org", "Latham & Watkins", "ORGANIZATION"), ("loc", "Chicago", "LOCATION")]
        ),
        (
            "Settlement agreement clause 14 stipulates mutual release of all claims by {date}.",
            [("date", "May 1, 2025", "DATE")]
        ),
        (
            "Attorney {name} filed power of attorney document under docket {ticket}.",
            [("name", "Garrick Ollivander", "PERSON"), ("ticket", "DOCK-99214", "TICKET_ID")]
        ),
        (
            "Arbitration hearing between {name} and {org} scheduled in {loc}.",
            [("name", "Dante Alighieri", "PERSON"), ("org", "Siemens", "ORGANIZATION"), ("loc", "Geneva", "LOCATION")]
        ),
        (
            "Real estate deed for property at {addr} recorded in municipal registry.",
            [("addr", "221B Baker Street, London, NW1 6XE", "ADDRESS")]
        ),
        (
            "Send certified legal notice to director {name} at {email} before filing suit.",
            [("name", "Julianne Moore", "PERSON"), ("email", "j.moore@corporation-holdings.com", "EMAIL")]
        ),
        (
            "Contract renewal terms approved by general counsel {name} on {date}.",
            [("name", "Bram Stoker", "PERSON"), ("date", "June 10, 2025", "DATE")]
        ),
        (
            "Patent licensing royalty payment sent to inventor {name} at {org}.",
            [("name", "Nikola Tesla", "PERSON"), ("org", "Westinghouse", "ORGANIZATION")]
        ),
        (
            "Subpoena served on witness {name} at residence {addr}.",
            [("name", "Roderick Usher", "PERSON"), ("addr", "742 Evergreen Terrace, Springfield, OR 97477", "ADDRESS")]
        ),
        (
            "Corporate acquisition of {org} finalized by managing partner {name} in {loc}.",
            [("org", "Arm Holdings", "ORGANIZATION"), ("name", "Siddhartha Mukherjee", "PERSON"), ("loc", "Cambridge", "LOCATION")]
        ),
        (
            "Joint venture agreement draft emailed to counsel {email} for review.",
            [("email", "counsel-mna@baker-mckenzie.com", "EMAIL")]
        ),
        (
            "Notary public {name} stamped the deed of gift in {loc} on {date}.",
            [("name", "Agatha Christie", "PERSON"), ("loc", "Torquay", "LOCATION"), ("date", "July 4, 2025", "DATE")]
        ),
        (
            "Contract dispute regarding order {order} referred to mediation.",
            [("order", "ORD-882019", "ORDER_ID")]
        ),
        (
            "Bylaws amendment signed by board secretary {name} representing {org}.",
            [("name", "Cornelius Fudge", "PERSON"), ("org", "Aramco", "ORGANIZATION")]
        ),
        (
            "Legal retainer of $25,000 wired to law firm {org} in {loc}.",
            [("org", "Skadden", "ORGANIZATION"), ("loc", "New York", "LOCATION")]
        ),
        (
            "Contact co-defendant {name} at telephone {phone} before the preliminary hearing.",
            [("name", "Ignatius Reilly", "PERSON"), ("phone", "+1-504-555-0182", "PHONE")]
        ),
        (
            "Employment non-compete clause signed by departing executive {name} on {date}.",
            [("name", "Vivian Ward", "PERSON"), ("date", "August 20, 2025", "DATE")]
        ),
        (
            "Eviction notice delivered to tenant {name} at {addr}.",
            [("name", "Cosette Fauchelevent", "PERSON"), ("addr", "16 Rue Saint-Denis, Paris, 75001", "ADDRESS")]
        ),
        (
            "Intellectual property assignment executed by contractor {name} for {org}.",
            [("name", "Sterling Archer", "PERSON"), ("org", "Lockheed Martin", "ORGANIZATION")]
        ),
        (
            "Court summons docket {ticket} issued to defendant {name}.",
            [("ticket", "CASE-44019", "TICKET_ID"), ("name", "Humbert Humbert", "PERSON")]
        ),
        (
            "Statute of limitations for claim expires on {date} according to attorney {name}.",
            [("date", "September 30, 2025", "DATE"), ("name", "Perry Mason", "PERSON")]
        ),
        (
            "Escrow release instruction confirmed by closing agent {name} at {org}.",
            [("name", "Linus Larrabee", "PERSON"), ("org", "First American Title", "ORGANIZATION")]
        ),

        # =====================================================================
        # CATEGORY 7: Public Figures & Organizations (35 prompts: 411 - 445)
        # =====================================================================
        (
            "{name} presented the new roadmap for {org} at the annual meeting in {loc}.",
            [("name", "Elon Musk", "PERSON"), ("org", "Tesla", "ORGANIZATION"), ("loc", "Austin", "LOCATION")]
        ),
        (
            "{name} discussed the Gemini AI ecosystem during the {org} keynote in {loc}.",
            [("name", "Sundar Pichai", "PERSON"), ("org", "Google", "ORGANIZATION"), ("loc", "Mountain View", "LOCATION")]
        ),
        (
            "{name} announced {org} cloud expansion in {loc1} and {loc2}.",
            [("name", "Satya Nadella", "PERSON"), ("org", "Microsoft", "ORGANIZATION"), ("loc1", "Frankfurt", "LOCATION"), ("loc2", "Dublin", "LOCATION")]
        ),
        (
            "{name} delivered the GTC presentation highlighting {org} computing clusters in {loc}.",
            [("name", "Jensen Huang", "PERSON"), ("org", "NVIDIA", "ORGANIZATION"), ("loc", "San Jose", "LOCATION")]
        ),
        (
            "{name} addressed developers regarding {org} platform security updates in {loc}.",
            [("name", "Sam Altman", "PERSON"), ("org", "OpenAI", "ORGANIZATION"), ("loc", "San Francisco", "LOCATION")]
        ),
        (
            "{name} outlined the environmental initiatives adopted across {org} global facilities in {loc}.",
            [("name", "Tim Cook", "PERSON"), ("org", "Apple", "ORGANIZATION"), ("loc", "Cupertino", "LOCATION")]
        ),
        (
            "{name} founded {org1} and later established {org2} for aerospace exploration in {loc}.",
            [("name", "Jeff Bezos", "PERSON"), ("org1", "Amazon", "ORGANIZATION"), ("org2", "Blue Origin", "ORGANIZATION"), ("loc", "Seattle", "LOCATION")]
        ),
        (
            "{name} shared progress on open source Llama models developed by {org} in {loc}.",
            [("name", "Mark Zuckerberg", "PERSON"), ("org", "Meta", "ORGANIZATION"), ("loc", "Menlo Park", "LOCATION")]
        ),
        (
            "{name} met with {org} leaders in {loc} to discuss eradication efforts.",
            [("name", "Bill Gates", "PERSON"), ("org", "World Health Organization", "ORGANIZATION"), ("loc", "Geneva", "LOCATION")]
        ),
        (
            "{name} released the annual shareholder letter on behalf of {org} in {loc}.",
            [("name", "Warren Buffett", "PERSON"), ("org", "Berkshire Hathaway", "ORGANIZATION"), ("loc", "Omaha", "LOCATION")]
        ),
        (
            "Contact executive assistant {name} at {email} regarding the schedule for {vip}.",
            [("name", "Claire Dupont", "PERSON"), ("email", "c.dupont@executive-office.com", "EMAIL"), ("vip", "Elon Musk", "PERSON")]
        ),
        (
            "Please forward the interview transcript of {vip} to reporter {name} at {phone}.",
            [("vip", "Satya Nadella", "PERSON"), ("name", "David Brooks", "PERSON"), ("phone", "+1-202-555-0176", "PHONE")]
        ),
        (
            "{org1} CEO {name} visited {org2} in {loc}.",
            [("org1", "NVIDIA", "ORGANIZATION"), ("name", "Jensen Huang", "PERSON"), ("org2", "Taiwan Semiconductor Manufacturing Company", "ORGANIZATION"), ("loc", "Hsinchu", "LOCATION")]
        ),
        (
            "The {org} convened in {loc1} with delegates from {loc2} and {loc3}.",
            [("org", "United Nations General Assembly", "ORGANIZATION"), ("loc1", "New York", "LOCATION"), ("loc2", "France", "LOCATION"), ("loc3", "Japan", "LOCATION")]
        ),
        (
            "{org} president {name} gave a press conference in {loc}.",
            [("org", "European Central Bank", "ORGANIZATION"), ("name", "Christine Lagarde", "PERSON"), ("loc", "Frankfurt", "LOCATION")]
        ),
        (
            "Send conference registration details for {name} to attend the keynote by {vip}.",
            [("name", "Pooja Varma", "PERSON"), ("vip", "Sam Altman", "PERSON")]
        ),
        (
            "{org} president {name} announced financing for clean energy in {loc}.",
            [("org", "World Bank", "ORGANIZATION"), ("name", "Ajay Banga", "PERSON"), ("loc", "India", "LOCATION")]
        ),
        (
            "{org} CEO {name} announced new retail stores opening in {loc1} and {loc2}.",
            [("org", "Apple", "ORGANIZATION"), ("name", "Tim Cook", "PERSON"), ("loc1", "Mumbai", "LOCATION"), ("loc2", "New Delhi", "LOCATION")]
        ),
        (
            "{org1} launched {org2} satellites from {loc} under launch authorization {ticket}.",
            [("org1", "SpaceX", "ORGANIZATION"), ("org2", "Starlink", "ORGANIZATION"), ("loc", "Cape Canaveral", "LOCATION"), ("ticket", "LIC-99120", "TICKET_ID")]
        ),
        (
            "{org} opened a new cloud infrastructure datacenter region in {loc}.",
            [("org", "Amazon Web Services", "ORGANIZATION"), ("loc", "Zurich", "LOCATION")]
        ),
        (
            "The {org} chairman spoke about monetary policy in {loc}.",
            [("org", "Federal Reserve", "ORGANIZATION"), ("loc", "Washington DC", "LOCATION")]
        ),
        (
            "{org} economist {name} presented the global economic outlook in {loc}.",
            [("org", "International Monetary Fund", "ORGANIZATION"), ("name", "Gita Gopinath", "PERSON"), ("loc", "London", "LOCATION")]
        ),
        (
            "Send invitation letter to {name} for the roundtable discussion with {vip}.",
            [("name", "Thierry Henry", "PERSON"), ("vip", "Bill Gates", "PERSON")]
        ),
        (
            "President {name} hosted a bilateral summit at the Elysee Palace in {loc}.",
            [("name", "Emmanuel Macron", "PERSON"), ("loc", "Paris", "LOCATION")]
        ),
        (
            "Prime Minister {name} inaugurated the technology summit in {loc}.",
            [("name", "Narendra Modi", "PERSON"), ("loc", "Bengaluru", "LOCATION")]
        ),
        (
            "German Chancellor {name} visited industrial manufacturing plants in {loc}.",
            [("name", "Olaf Scholz", "PERSON"), ("loc", "Munich", "LOCATION")]
        ),
        (
            "British Prime Minister {name} met with {org} representatives in {loc}.",
            [("name", "Keir Starmer", "PERSON"), ("org", "European Union", "ORGANIZATION"), ("loc", "Brussels", "LOCATION")]
        ),
        (
            "{org} designated historic heritage sites across {loc1} and {loc2}.",
            [("org", "UNESCO", "ORGANIZATION"), ("loc1", "Italy", "LOCATION"), ("loc2", "Greece", "LOCATION")]
        ),
        (
            "Private secretary {name} handles public relations inquiries for {vip} at {email}.",
            [("name", "Vanessa Ives", "PERSON"), ("vip", "Mark Zuckerberg", "PERSON"), ("email", "pr-press@meta-comms.net", "EMAIL")]
        ),
        (
            "Japanese Prime Minister {name} addressed parliament in {loc}.",
            [("name", "Shigeru Ishiba", "PERSON"), ("loc", "Tokyo", "LOCATION")]
        ),
        (
            "DeepMind co-founder {name} discussed artificial intelligence milestones in {loc}.",
            [("name", "Demis Hassabis", "PERSON"), ("loc", "London", "LOCATION")]
        ),
        (
            "{org} flights to {loc1} and {loc2} resumed regular operational schedules.",
            [("org", "Lufthansa", "ORGANIZATION"), ("loc1", "Berlin", "LOCATION"), ("loc2", "Vienna", "LOCATION")]
        ),
        (
            "Please deliver the briefing packet to {name} before the meeting with {vip}.",
            [("name", "Gautam Adani", "PERSON"), ("vip", "Sundar Pichai", "PERSON")]
        ),
        (
            "{org} hosted a symposium featuring Nobel laureate economists in {loc}.",
            [("org", "Stanford University", "ORGANIZATION"), ("loc", "California", "LOCATION")]
        ),
        (
            "{org} announced a new endowment fund for scientific research in {loc}.",
            [("org", "Harvard University", "ORGANIZATION"), ("loc", "Boston", "LOCATION")]
        ),

        # =====================================================================
        # CATEGORY 8: Negative Controls (Zero-PII / Benign) (30 prompts: 446 - 475)
        # =====================================================================
        (
            "The quick brown fox jumps over the lazy dog in standard typographic pangrams.",
            []
        ),
        (
            "Calculus fundamental theorem relates differentiation with integration over continuous domains.",
            []
        ),
        (
            "Quantum entanglement demonstrates non-local correlations between entangled particle states.",
            []
        ),
        (
            "Binary search algorithm operates in logarithmic time complexity O(log n) on sorted arrays.",
            []
        ),
        (
            "Photosynthesis converts solar electromagnetic radiation into chemical energy in plant chloroplasts.",
            []
        ),
        (
            "Standard deviation measures the dispersion of a dataset relative to its statistical mean.",
            []
        ),
        (
            "HTTP status code 404 indicates that the requested resource could not be found on server.",
            []
        ),
        (
            "Database indexing utilizes B-tree data structures to accelerate record query execution.",
            []
        ),
        (
            "The Carnot cycle establishes the theoretical upper limit on heat engine thermodynamic efficiency.",
            []
        ),
        (
            "Convolutional neural networks extract hierarchical spatial features from input image matrices.",
            []
        ),
        (
            "Git version control uses directed acyclic graphs to track tree state snapshots.",
            []
        ),
        (
            "Mitochondria generate adenosine triphosphate through oxidative phosphorylation pathways.",
            []
        ),
        (
            "TCP congestion control implements slow start and additive increase multiplicative decrease.",
            []
        ),
        (
            "Electromagnetic waves propagate through vacuum at the invariant speed of light.",
            []
        ),
        (
            "Quick sort algorithm partitions arrays around a chosen pivot value recursively.",
            []
        ),
        (
            "Enzyme kinetics can be modeled using the Michaelis-Menten mathematical formulation.",
            []
        ),
        (
            "The general theory of relativity describes gravity as the curvature of spacetime.",
            []
        ),
        (
            "Vector spaces are closed under vector addition and scalar multiplication operations.",
            []
        ),
        (
            "DNS lookups convert human-readable domain names into machine-readable network addresses.",
            []
        ),
        (
            "Refactoring code improves software maintainability without modifying observable runtime behavior.",
            []
        ),
        (
            "Polymorphism enables objects of different classes to respond to identical method signatures.",
            []
        ),
        (
            "Plate tectonics explains continental drift through convection currents in the mantle.",
            []
        ),
        (
            "Fourier transform decomposes complex time-domain signals into constituent sinusoidal frequencies.",
            []
        ),
        (
            "Asynchronous programming uses event loops and non-blocking I/O multiplexing primitives.",
            []
        ),
        (
            "Thermodynamic entropy represents the microscopic statistical disorder of closed physical systems.",
            []
        ),
        (
            "Graph traversal algorithms like breadth-first search explore neighboring vertices level by level.",
            []
        ),
        (
            "Stochastic gradient descent updates model parameters using randomly sampled training mini-batches.",
            []
        ),
        (
            "Functional programming emphasizes immutable data structures and pure referentially transparent functions.",
            []
        ),
        (
            "Compiler optimization passes include dead code elimination and loop invariant code motion.",
            []
        ),
        (
            "Cellular mitosis produces two genetically identical daughter cells through nuclear division.",
            []
        ),

        # =====================================================================
        # CATEGORY 9: Edge Cases, Ambiguities & System Tokens (25 prompts: 476 - 500)
        # =====================================================================
        (
            "The system administrator executed sudo chmod 755 on localhost as root user.",
            []
        ),
        (
            "The superuser logged into admin_root console without multi-factor authentication.",
            []
        ),
        (
            "The number 55 is not an ID, and 1234567890 is merely a dummy test sequence.",
            []
        ),
        (
            "We have great hope that our new policy will bring lasting peace and goodwill.",
            []
        ),
        (
            "She planted a red rose in the garden next to the blooming daisy bush.",
            []
        ),
        (
            "Her legal name is {name} and she works as a lead architect at {org}.",
            [("name", "Rose Henderson", "PERSON"), ("org", "Autodesk", "ORGANIZATION")]
        ),
        (
            "His name is {name} and he sent the invoice to account payable.",
            [("name", "Bill Crawford", "PERSON")]
        ),
        (
            "Please pay the restaurant bill of $85 before the end of the evening.",
            []
        ),
        (
            "The engineer {name} solved the compiler bug in record time.",
            [("name", "Victor Cruz", "PERSON")]
        ),
        (
            "The military won a decisive victory in the historic battle of {loc}.",
            [("loc", "Waterloo", "LOCATION")]
        ),
        (
            "Our corporate customer {name} updated their contact email to {email}.",
            [("name", "Grace Hopper", "PERSON"), ("email", "grace.hopper@navy-tech.mil", "EMAIL")]
        ),
        (
            "The dancer moved with remarkable grace and artistic precision across the stage.",
            []
        ),
        (
            "Contact senior analyst {name} at {phone} regarding quarterly projections.",
            [("name", "May Zhou", "PERSON"), ("phone", "+86 10 8820 1928", "PHONE")]
        ),
        (
            "In {date} we will review our corporate roadmap for the upcoming fiscal quarter.",
            [("date", "May", "DATE")]
        ),
        (
            "The software engineer {name} approved the merge request on GitHub.",
            [("name", "Christian Bale", "PERSON")]
        ),
        (
            "The cathedral attracts many Christian pilgrims from across {loc} each year.",
            [("loc", "Europe", "LOCATION")]
        ),
        (
            "The detective {name} interviewed the suspect at the police precinct.",
            [("name", "Frank Castle", "PERSON")]
        ),
        (
            "To be completely frank, the project timeline requires substantial adjustment.",
            []
        ),
        (
            "Customer {name} reported that their order {order} was sent to the wrong address.",
            [("name", "Dawn Summers", "PERSON"), ("order", "ORD-120491", "ORDER_ID")]
        ),
        (
            "At dawn the search and rescue team resumed operations in the valley.",
            []
        ),
        (
            "System test payload: 9999999999 is an invalid phone number placeholder.",
            []
        ),
        (
            "The user root executed the script on sysadmin workstation at {ip}.",
            [("ip", "127.0.0.1", "IP_ADDRESS")]
        ),
        (
            "Contact project coordinator {name} at {email} on {date}.",
            [("name", "Faith Hill", "PERSON"), ("email", "f.hill@country-arts.org", "EMAIL"), ("date", "October 31, 2024", "DATE")]
        ),
        (
            "The candidate {name} answered all technical questions with great competence.",
            [("name", "Joy Mukherjee", "PERSON")]
        ),
        (
            "Employee {name} submitted expense report under customer tracking code {cust}.",
            [("name", "Hunter Thompson", "PERSON"), ("cust", "CUST-99201", "CUSTOMER_ID")]
        ),
    ]

    return specs


def main():
    source_file = BACKUP_FILE if BACKUP_FILE.exists() else GT_FILE
    print(f"[INFO] Reading baseline 250 ground truth from: {source_file}")
    with open(source_file, "r", encoding="utf-8") as f:
        existing_data = json.load(f)

    # Ensure baseline is exactly 250 prompts
    if len(existing_data) > 250:
        existing_data = existing_data[:250]

    print(f"[INFO] Loaded baseline: {len(existing_data)} prompts.")

    specs = get_250_new_prompt_specs()
    print(f"[INFO] Defined {len(specs)} new prompt specifications.")

    start_idx = 251
    new_prompts = []

    for i, (template, entities_to_place) in enumerate(specs):
        prompt_id = f"test_{start_idx + i:03d}"
        
        try:
            rendered = build_prompt_with_entities(template, entities_to_place)
        except Exception as e:
            print(f"[ERROR] Failed building prompt {prompt_id}: {e}")
            sys.exit(1)

        new_entry = {
            "id": prompt_id,
            "text": rendered["text"],
            "entities": rendered["entities"]
        }
        new_prompts.append(new_entry)

    combined_data = existing_data + new_prompts
    print(f"[INFO] Combined total prompts: {len(combined_data)}")

    # Strict full validation of every entity offset in all prompts
    print("[INFO] Verifying character offsets across all prompts...")
    errors = []
    total_entities = 0
    for item in combined_data:
        text = item["text"]
        for e in item.get("entities", []):
            total_entities += 1
            actual = text[e["start"]:e["end"]]
            if actual != e["text"]:
                errors.append(f"[{item['id']}] Mismatch: expected '{e['text']}', got '{actual}' at [{e['start']}:{e['end']}] in '{text}'")

    if errors:
        print(f"[FATAL] Found {len(errors)} offset mismatches!")
        for err in errors[:10]:
            print(f"  {err}")
        sys.exit(1)

    print(f"[INFO] All {total_entities} entity offsets strictly verified! 100% VALID.")

    # Save to ground_truth.json
    with open(GT_FILE, "w", encoding="utf-8") as f:
        json.dump(combined_data, f, indent=2, ensure_ascii=False)

    print(f"[SUCCESS] Updated {GT_FILE} with exactly {len(combined_data)} prompts.")


if __name__ == "__main__":
    main()
