"""
Generates 50 realistic legal PDF documents and uploads them to the RAG system.
"""
import os
import requests
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.colors import HexColor
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable, Table, TableStyle
from reportlab.lib import colors

API = "http://localhost:8000"
FOLDER = "Legal Documents"
OUT_DIR = Path("/tmp/legal_docs")
OUT_DIR.mkdir(exist_ok=True)

# ── Styles ────────────────────────────────────────────────────────────────────

def make_styles():
    base = getSampleStyleSheet()
    accent = HexColor("#1a3a5c")
    return {
        "title": ParagraphStyle("title", parent=base["Heading1"],
            fontSize=16, textColor=accent, spaceAfter=4, leading=20),
        "subtitle": ParagraphStyle("subtitle", parent=base["Normal"],
            fontSize=10, textColor=HexColor("#555555"), spaceAfter=12, leading=14),
        "section": ParagraphStyle("section", parent=base["Heading2"],
            fontSize=11, textColor=accent, spaceBefore=14, spaceAfter=4, leading=14),
        "body": ParagraphStyle("body", parent=base["Normal"],
            fontSize=9.5, leading=15, spaceAfter=6),
        "small": ParagraphStyle("small", parent=base["Normal"],
            fontSize=8.5, textColor=HexColor("#777777"), leading=12),
    }

def build_pdf(path, title, doc_type, parties, date, sections):
    doc = SimpleDocTemplate(str(path), pagesize=A4,
        leftMargin=2.5*cm, rightMargin=2.5*cm,
        topMargin=2.5*cm, bottomMargin=2.5*cm)
    S = make_styles()
    story = []

    # Header
    story.append(Paragraph(title, S["title"]))
    story.append(Paragraph(f"{doc_type} &nbsp;·&nbsp; {date}", S["subtitle"]))
    story.append(HRFlowable(width="100%", thickness=1.5,
        color=HexColor("#1a3a5c"), spaceAfter=10))

    # Parties table
    if parties:
        rows = [["PARTIES TO THIS AGREEMENT", ""]]
        for role, name in parties:
            rows.append([role, name])
        t = Table(rows, colWidths=[4*cm, 12*cm])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), HexColor("#1a3a5c")),
            ("TEXTCOLOR",  (0,0), (-1,0), colors.white),
            ("FONTSIZE",   (0,0), (-1,0), 8),
            ("FONTNAME",   (0,0), (-1,0), "Helvetica-Bold"),
            ("BACKGROUND", (0,1), (0,-1), HexColor("#eef2f7")),
            ("FONTSIZE",   (0,1), (-1,-1), 9),
            ("FONTNAME",   (0,1), (0,-1), "Helvetica-Bold"),
            ("GRID",       (0,0), (-1,-1), 0.5, HexColor("#cccccc")),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, HexColor("#f8f9fb")]),
            ("LEFTPADDING",  (0,0), (-1,-1), 6),
            ("RIGHTPADDING", (0,0), (-1,-1), 6),
            ("TOPPADDING",   (0,0), (-1,-1), 4),
            ("BOTTOMPADDING",(0,0), (-1,-1), 4),
        ]))
        story.append(t)
        story.append(Spacer(1, 12))

    # Sections
    for i, (heading, text) in enumerate(sections, 1):
        story.append(Paragraph(f"{i}. {heading}", S["section"]))
        for para in text:
            story.append(Paragraph(para, S["body"]))

    # Footer
    story.append(Spacer(1, 20))
    story.append(HRFlowable(width="100%", thickness=0.5,
        color=HexColor("#cccccc"), spaceAfter=6))
    story.append(Paragraph(
        f"This document was prepared on {date}. All rights reserved. "
        "This document is legally binding upon execution by all parties.",
        S["small"]))

    doc.build(story)


# ── Document definitions ──────────────────────────────────────────────────────

DOCS = [
  # ── Employment ──
  ("Employment_Agreement_Software_Engineer", "Employment Agreement — Software Engineer",
   "Employment Contract", "March 15, 2024",
   [("Employer", "TechCorp Solutions Inc."), ("Employee", "Alex Johnson")],
   [
     ("Position and Duties", ["The Employee is hired as Senior Software Engineer reporting to the VP of Engineering.",
       "Duties include designing, developing, and maintaining software systems, conducting code reviews, and mentoring junior developers."]),
     ("Compensation", ["Base salary: $145,000 per year, paid bi-weekly.",
       "Annual performance bonus up to 15% of base salary, subject to company and individual performance targets.",
       "Stock option grant of 2,000 shares vesting over 4 years with 1-year cliff."]),
     ("Working Hours", ["Standard working hours are 40 hours per week, Monday through Friday.",
       "Remote work is permitted up to 3 days per week subject to manager approval.",
       "Overtime may be required during critical project phases and will be compensated in accordance with applicable law."]),
     ("Confidentiality", ["Employee agrees to maintain strict confidentiality of all proprietary information, trade secrets, and business strategies.",
       "This obligation survives termination of employment for a period of 3 years."]),
     ("Termination", ["Either party may terminate this agreement with 30 days written notice.",
       "The Company may terminate without notice for cause, including gross misconduct, fraud, or material breach of this agreement."]),
   ]),

  ("Non_Disclosure_Agreement_Partnership", "Non-Disclosure Agreement",
   "NDA / Confidentiality Agreement", "January 10, 2024",
   [("Disclosing Party", "AlphaTech Ventures LLC"), ("Receiving Party", "Beta Solutions GmbH")],
   [
     ("Purpose", ["The parties wish to explore a potential business partnership and may disclose confidential information to each other for evaluation purposes only."]),
     ("Definition of Confidential Information", ["Confidential Information means any data or information, oral or written, that relates to business plans, financial data, technical specifications, customer lists, pricing strategies, or any other proprietary information.",
       "Information is not confidential if it was already publicly available, independently developed, or lawfully received from a third party."]),
     ("Obligations", ["The Receiving Party shall: (a) hold the Confidential Information in strict confidence; (b) not disclose it to any third party without prior written consent; (c) use it solely for the Purpose stated above.",
       "The Receiving Party shall apply at least the same degree of care as it uses for its own confidential information, but no less than reasonable care."]),
     ("Term", ["This Agreement is effective for 2 years from the date of signing.",
       "Upon expiration or termination, the Receiving Party shall promptly return or destroy all Confidential Information."]),
     ("Remedies", ["The parties acknowledge that breach of this Agreement would cause irreparable harm for which monetary damages would be inadequate.",
       "The Disclosing Party is entitled to seek injunctive relief in addition to other legal remedies."]),
   ]),

  ("Independent_Contractor_Agreement_Design", "Independent Contractor Agreement",
   "Service Contract", "February 1, 2024",
   [("Client", "Meridian Marketing Group"), ("Contractor", "Jane Smith Design Studio")],
   [
     ("Scope of Work", ["Contractor will provide graphic design services including brand identity, web design mockups, and marketing materials.",
       "Specific deliverables are defined in Exhibit A attached hereto."]),
     ("Payment Terms", ["Client will pay Contractor $8,500 upon project completion.",
       "Invoices are due within 15 days of receipt. Late payments accrue interest at 1.5% per month."]),
     ("Intellectual Property", ["All work product created under this Agreement shall be work-for-hire and become the exclusive property of Client upon full payment.",
       "Contractor retains the right to display the work in their portfolio unless Client requests otherwise in writing."]),
     ("Independent Contractor Status", ["Contractor is an independent contractor, not an employee.",
       "Contractor is responsible for all taxes, insurance, and benefits. Client will not withhold taxes or provide benefits."]),
     ("Termination", ["Either party may terminate this Agreement with 14 days written notice.",
       "Client shall pay for all work completed up to the termination date."]),
   ]),

  ("Lease_Agreement_Commercial_Office", "Commercial Lease Agreement",
   "Real Estate Lease", "March 1, 2024",
   [("Landlord", "Urban Property Holdings Ltd."), ("Tenant", "Nexus Startup Inc.")],
   [
     ("Premises", ["Landlord leases to Tenant the office space located at Suite 400, 1200 Commerce Blvd, totaling 2,400 square feet."]),
     ("Term and Rent", ["Lease term: 24 months commencing April 1, 2024.",
       "Monthly rent: $4,800, due on the first of each month. A security deposit of $9,600 is required upon signing."]),
     ("Use of Premises", ["Tenant shall use the premises solely for general office and administrative purposes.",
       "Tenant shall not use the premises for any unlawful purpose or in any manner that creates a nuisance."]),
     ("Maintenance", ["Tenant is responsible for interior maintenance and minor repairs under $500.",
       "Landlord is responsible for structural repairs, HVAC systems, and common area maintenance."]),
     ("Renewal Option", ["Tenant has the option to renew for one additional 12-month term by providing 60 days written notice before lease expiration.",
       "Renewal rent shall not exceed 105% of the current monthly rate."]),
   ]),

  ("Software_License_Agreement_Enterprise", "Software License Agreement",
   "Enterprise License", "January 20, 2024",
   [("Licensor", "CloudBase Technologies Inc."), ("Licensee", "Fortuna Retail Corp.")],
   [
     ("Grant of License", ["Licensor grants Licensee a non-exclusive, non-transferable license to use the CloudBase Analytics Platform for internal business purposes.",
       "The license covers up to 250 concurrent users across all Licensee's locations."]),
     ("Restrictions", ["Licensee may not: (a) sublicense or resell the software; (b) reverse engineer or decompile it; (c) use it to develop competing products.",
       "Licensee may create one backup copy solely for archival purposes."]),
     ("Fees", ["Annual license fee: $72,000, invoiced annually in advance.",
       "A 30-day free trial period is provided before the first invoice."]),
     ("Support and Updates", ["Licensor will provide standard support during business hours and software updates at no additional charge.",
       "Critical security patches will be delivered within 48 hours of release."]),
     ("Term and Termination", ["This agreement is effective for 3 years and auto-renews unless either party provides 90 days notice.",
       "Upon termination, Licensee must uninstall all copies of the software and certify compliance in writing."]),
   ]),

  ("Service_Level_Agreement_Cloud_Hosting", "Service Level Agreement",
   "SLA — Cloud Infrastructure", "February 15, 2024",
   [("Service Provider", "NovaClouds Ltd."), ("Customer", "DataFlow Analytics Inc.")],
   [
     ("Service Description", ["Provider will supply managed cloud hosting including virtual servers, load balancing, automated backups, and 24/7 monitoring."]),
     ("Uptime Guarantee", ["Provider guarantees 99.9% monthly uptime for all production services.",
       "Scheduled maintenance windows, not exceeding 4 hours per month, are excluded from uptime calculations."]),
     ("Response Times", ["Critical incidents (service down): response within 15 minutes, resolution target 2 hours.",
       "High priority: response within 1 hour. Medium priority: response within 4 business hours."]),
     ("Service Credits", ["For each 1% of uptime below the guaranteed level, Customer receives a credit of 5% of the monthly fee.",
       "Maximum monthly credit is 30% of the monthly fee. Credits are applied to the next invoice."]),
     ("Exclusions", ["The SLA does not apply to outages caused by Customer's actions, third-party providers, or force majeure events."]),
   ]),

  ("Privacy_Policy_eCommerce", "Privacy Policy",
   "Data Protection Policy", "January 1, 2024",
   [],
   [
     ("Data We Collect", ["We collect information you provide directly: name, email, billing address, and payment details.",
       "We automatically collect usage data including IP address, browser type, pages visited, and purchase history.",
       "We may receive data from third-party partners such as advertising networks and analytics providers."]),
     ("How We Use Your Data", ["To process orders, send confirmations, and provide customer support.",
       "To personalize your shopping experience and send promotional offers (with your consent).",
       "To comply with legal obligations and prevent fraud."]),
     ("Data Sharing", ["We do not sell your personal data to third parties.",
       "We share data with payment processors, shipping carriers, and cloud service providers solely as necessary to fulfill your orders.",
       "We may disclose data if required by law or to protect our legal rights."]),
     ("Your Rights", ["You have the right to access, correct, or delete your personal data at any time.",
       "To exercise these rights, contact our Data Protection Officer at privacy@shop.com.",
       "You may also lodge a complaint with your local data protection authority."]),
     ("Data Retention", ["We retain order data for 7 years for tax and legal compliance.",
       "Marketing data is deleted within 12 months of your last interaction or upon withdrawal of consent."]),
   ]),

  ("Terms_of_Service_SaaS_Platform", "Terms of Service",
   "SaaS Platform Terms", "January 1, 2024",
   [],
   [
     ("Acceptance of Terms", ["By creating an account or using our platform, you agree to these Terms of Service.",
       "If you do not agree, do not use our services."]),
     ("User Accounts", ["You must provide accurate information when creating an account.",
       "You are responsible for maintaining the confidentiality of your login credentials.",
       "You must notify us immediately of any unauthorized access to your account."]),
     ("Acceptable Use", ["You may not use the platform to transmit spam, malware, or illegal content.",
       "You may not attempt to gain unauthorized access to other users' data or our infrastructure.",
       "Violation of these rules may result in immediate account termination without refund."]),
     ("Fees and Billing", ["Subscription fees are billed monthly in advance. All fees are non-refundable.",
       "We reserve the right to change pricing with 30 days notice. Continued use after the notice period constitutes acceptance."]),
     ("Limitation of Liability", ["Our liability is limited to the amount you paid in the 3 months preceding the claim.",
       "We are not liable for indirect, incidental, or consequential damages."]),
   ]),

  ("Partnership_Agreement_Joint_Venture", "Partnership Agreement",
   "Joint Venture Agreement", "March 1, 2024",
   [("Partner A", "Sunrise Capital Group"), ("Partner B", "Horizon Development LLC")],
   [
     ("Formation", ["The parties hereby form a joint venture for the purpose of acquiring, developing, and selling residential properties.",
       "The joint venture shall operate under the name 'Sunrise-Horizon Realty Partners'."]),
     ("Capital Contributions", ["Partner A contributes $2,000,000 cash and holds a 60% interest.",
       "Partner B contributes project management expertise and land rights valued at $1,333,333 and holds a 40% interest."]),
     ("Management", ["Major decisions require unanimous consent of both Partners.",
       "Day-to-day operations are managed by a Management Committee with equal representation.",
       "Neither Partner may encumber venture assets without the other's written consent."]),
     ("Profit and Loss Distribution", ["Net profits and losses are allocated in proportion to ownership interests.",
       "Distributions are made quarterly after retaining sufficient reserves for operations."]),
     ("Dissolution", ["The joint venture dissolves upon completion of all projects, mutual agreement, or court order.",
       "Upon dissolution, assets are liquidated, debts paid, and remaining proceeds distributed according to ownership interests."]),
   ]),

  ("Intellectual_Property_Assignment_Agreement", "Intellectual Property Assignment Agreement",
   "IP Assignment", "February 20, 2024",
   [("Assignor", "Dr. Michael Chen"), ("Assignee", "BioTech Innovations Corp.")],
   [
     ("Assignment", ["Assignor hereby irrevocably assigns to Assignee all rights, title, and interest in and to the invention titled 'Automated Protein Folding Analysis System' (the 'Invention').",
       "The assignment includes all patents, patent applications, trade secrets, and know-how related to the Invention."]),
     ("Consideration", ["In consideration of this assignment, Assignee shall pay Assignor $50,000 upon execution plus a royalty of 2% of net sales for 5 years."]),
     ("Representations", ["Assignor represents that: (a) the Invention is original; (b) no other party has rights to it; (c) it does not infringe any third-party rights."]),
     ("Cooperation", ["Assignor will execute all documents and provide all assistance reasonably necessary for Assignee to obtain and enforce patent protection worldwide."]),
     ("Governing Law", ["This Agreement is governed by the laws of the State of Delaware."]),
   ]),

  ("Loan_Agreement_Business", "Business Loan Agreement",
   "Loan Agreement", "February 28, 2024",
   [("Lender", "First Community Bank N.A."), ("Borrower", "Green Leaf Restaurant LLC")],
   [
     ("Loan Amount", ["Lender agrees to lend Borrower the principal sum of $150,000 ('Loan')."]),
     ("Interest Rate", ["The Loan bears interest at a fixed rate of 7.5% per annum.",
       "Interest accrues daily on the outstanding principal balance."]),
     ("Repayment", ["Borrower shall repay the Loan in 60 equal monthly installments of $3,004.57 each.",
       "First payment is due May 1, 2024. Payments are due on the first of each month thereafter."]),
     ("Prepayment", ["Borrower may prepay all or any portion of the Loan at any time without penalty.",
       "Prepayments shall be applied first to accrued interest, then to principal."]),
     ("Events of Default", ["Default occurs if: (a) payment is 15 days past due; (b) Borrower becomes insolvent; (c) any representation was materially false.",
       "Upon default, the full outstanding balance becomes immediately due and payable."]),
   ]),

  ("Consulting_Agreement_Management", "Consulting Services Agreement",
   "Consulting Contract", "January 15, 2024",
   [("Client", "Pinnacle Manufacturing Inc."), ("Consultant", "Strategic Outcomes LLC")],
   [
     ("Services", ["Consultant will provide operational efficiency consulting, including process analysis, workflow redesign, and implementation support.",
       "Services will be delivered over a 6-month engagement beginning February 1, 2024."]),
     ("Compensation", ["Client pays $25,000 per month for up to 80 consulting hours.",
       "Additional hours beyond 80 per month are billed at $350 per hour.",
       "Reasonable travel expenses are reimbursed upon submission of receipts."]),
     ("Deliverables", ["Month 1-2: Current state assessment and gap analysis report.",
       "Month 3-4: Redesigned process documentation and implementation roadmap.",
       "Month 5-6: Implementation support and final performance report."]),
     ("Confidentiality and Non-Compete", ["Consultant agrees to maintain confidentiality for 3 years post-engagement.",
       "Consultant will not solicit Client's employees for 12 months after engagement ends."]),
     ("Indemnification", ["Each party indemnifies the other against claims arising from its own negligence or willful misconduct."]),
   ]),

  ("Shareholder_Agreement_Startup", "Shareholder Agreement",
   "Corporate Governance", "March 10, 2024",
   [("Founder A", "Sarah Williams (40%)"), ("Founder B", "Tom Garcia (35%)"), ("Investor", "Seed Fund I LP (25%)")],
   [
     ("Share Classes", ["Class A shares carry one vote each and are held by Founders.",
       "Class B shares carry no votes and are held by Investor.",
       "All shares participate equally in dividends and liquidation proceeds."]),
     ("Board of Directors", ["The Board consists of 5 directors: 2 appointed by Founder A, 1 by Founder B, 1 by Investor, and 1 independent director.",
       "Board decisions require a majority vote. Major decisions require 4/5 approval."]),
     ("Transfer Restrictions", ["Shares may not be transferred without Board approval.",
       "Existing shareholders have a right of first refusal on any proposed transfer.",
       "A co-sale right allows shareholders to participate in any third-party sale."]),
     ("Anti-Dilution", ["In the event of a down round, Investor's shares are subject to broad-based weighted average anti-dilution protection."]),
     ("Liquidation Preference", ["Upon liquidation, Investor receives 1x its investment before any distribution to Founders.",
       "Remaining proceeds are distributed pro-rata to all shareholders."]),
   ]),

  ("Supply_Agreement_Manufacturing", "Supply Agreement",
   "Procurement Contract", "February 5, 2024",
   [("Supplier", "Pacific Components Co."), ("Buyer", "Atlas Electronics Ltd.")],
   [
     ("Products", ["Supplier agrees to supply electronic components as specified in Schedule A, including circuit boards, capacitors, and connectors."]),
     ("Pricing", ["Unit prices are fixed for 12 months per the attached Price List.",
       "After 12 months, prices may be adjusted with 60 days notice; Buyer may terminate if increase exceeds 5%."]),
     ("Delivery", ["Supplier shall deliver to Buyer's warehouse within 15 business days of each purchase order.",
       "Supplier bears risk of loss until delivery is accepted by Buyer."]),
     ("Quality Standards", ["All products must meet the ISO 9001 quality standards and Buyer's technical specifications.",
       "Buyer may reject non-conforming goods and require replacement within 10 business days at Supplier's cost."]),
     ("Warranty", ["Supplier warrants products are free from defects for 24 months from delivery.",
       "This warranty does not cover damage caused by improper use, storage, or modification."]),
   ]),

  ("Distribution_Agreement_International", "International Distribution Agreement",
   "Distribution Contract", "January 25, 2024",
   [("Manufacturer", "Nova Foods Inc."), ("Distributor", "EuroMarket Trading GmbH")],
   [
     ("Territory", ["Distributor has exclusive distribution rights for Germany, Austria, and Switzerland ('Territory').",
       "Manufacturer will not appoint other distributors or sell directly in the Territory during the agreement term."]),
     ("Minimum Purchase Obligations", ["Distributor commits to purchase at least $500,000 of products annually.",
       "Failure to meet minimums allows Manufacturer to convert the appointment to non-exclusive."]),
     ("Pricing and Payment", ["Products are priced per the Manufacturer's current export price list.",
       "Payment is due 30 days after invoice date via bank transfer."]),
     ("Marketing Obligations", ["Distributor will actively promote products and maintain a dedicated sales team.",
       "Distributor will spend at least 3% of annual purchases on local marketing."]),
     ("Term", ["Initial term is 3 years, auto-renewing for 1-year periods absent 90-day notice of non-renewal."]),
   ]),

  ("Data_Processing_Agreement_GDPR", "Data Processing Agreement",
   "GDPR Compliance — DPA", "January 1, 2024",
   [("Controller", "ShopFast Europe Ltd."), ("Processor", "CloudCRM Solutions Inc.")],
   [
     ("Subject Matter", ["Processor processes personal data on behalf of Controller for the purpose of customer relationship management."]),
     ("Nature of Processing", ["Processing activities include storing, organizing, retrieving, and deleting customer contact data and purchase history."]),
     ("Processor Obligations", ["Processor shall: (a) process data only on documented instructions from Controller; (b) ensure persons authorized to process data are bound by confidentiality; (c) implement appropriate technical and organizational security measures."]),
     ("Sub-processors", ["Processor may engage sub-processors listed in Annex B.",
       "Processor remains fully liable to Controller for sub-processors' performance.",
       "Controller will be notified of any changes to sub-processors 30 days in advance."]),
     ("Data Subject Rights", ["Processor shall assist Controller in responding to data subject requests within the timeframes required by applicable law."]),
   ]),

  ("Franchise_Agreement_Restaurant", "Franchise Agreement",
   "Franchise Contract", "March 5, 2024",
   [("Franchisor", "BurgerWorld International Inc."), ("Franchisee", "Local Foods LLC")],
   [
     ("Grant of Franchise", ["Franchisor grants Franchisee the right to operate a BurgerWorld restaurant at 555 Main Street, Springfield.",
       "This is a non-exclusive license limited to the specified location."]),
     ("Fees", ["Initial franchise fee: $45,000, payable upon signing.",
       "Ongoing royalty: 6% of monthly gross sales, due by the 15th of each month.",
       "Marketing contribution: 2% of monthly gross sales to the national advertising fund."]),
     ("Training", ["Franchisee and designated manager must complete the 3-week BurgerWorld Training Program before opening.",
       "Ongoing annual training of 3 days is required."]),
     ("Standards", ["Franchisee must comply with all BurgerWorld brand standards, including menu, decor, uniforms, and customer service protocols.",
       "Franchisor may conduct unannounced inspections. Failure to meet standards may result in termination."]),
     ("Term", ["Agreement term is 10 years with two 5-year renewal options, provided Franchisee is in good standing."]),
   ]),

  ("Settlement_Agreement_Dispute", "Settlement Agreement and Release",
   "Legal Settlement", "February 29, 2024",
   [("Claimant", "Marcus Thompson"), ("Respondent", "Delta Logistics Inc.")],
   [
     ("Background", ["Claimant filed a claim alleging breach of contract and damages arising from Respondent's failure to deliver goods on time.",
       "The parties desire to resolve this matter without further litigation."]),
     ("Settlement Payment", ["Respondent agrees to pay Claimant $32,500 within 10 business days of execution.",
       "Payment shall be made by wire transfer to the account specified by Claimant."]),
     ("Release", ["In consideration of the settlement payment, Claimant releases and discharges Respondent from all claims, known or unknown, arising out of or related to the contract dated August 1, 2023."]),
     ("No Admission", ["This settlement is not an admission of liability or wrongdoing by either party."]),
     ("Confidentiality", ["The parties agree to keep the terms of this settlement confidential, except as required by law or for tax purposes."]),
   ]),

  ("Vendor_Agreement_IT_Services", "IT Vendor Services Agreement",
   "Technology Services Contract", "January 30, 2024",
   [("Vendor", "TechSupport Pro LLC"), ("Customer", "Heritage Financial Group")],
   [
     ("Services", ["Vendor provides managed IT services including helpdesk support, network monitoring, cybersecurity management, and hardware procurement."]),
     ("Service Levels", ["Helpdesk response time: critical issues within 30 minutes, standard within 4 hours.",
       "Network uptime: 99.5% monthly guarantee excluding scheduled maintenance."]),
     ("Term and Fees", ["Initial term: 24 months. Monthly fee: $8,200 covering all services.",
       "Hardware is invoiced separately at Vendor's cost plus 12%."]),
     ("Security", ["Vendor will implement and maintain enterprise-grade firewalls, endpoint protection, and email security.",
       "Vendor will conduct quarterly vulnerability assessments and provide written reports."]),
     ("Data Handling", ["Vendor acknowledges it may access sensitive financial data and agrees to comply with all applicable data protection regulations.",
       "Vendor will not retain, copy, or use Customer data beyond what is necessary to perform the services."]),
   ]),

  ("Memorandum_of_Understanding_Collaboration", "Memorandum of Understanding",
   "MOU — Research Collaboration", "March 12, 2024",
   [("Institution A", "State University Research Institute"), ("Institution B", "PharmaTech R&D Center")],
   [
     ("Purpose", ["The parties agree to collaborate on research into novel drug delivery mechanisms, sharing resources, data, and personnel."]),
     ("Responsibilities", ["University provides laboratory facilities, graduate researchers, and computational resources.",
       "PharmaTech provides funding of $300,000 per year, proprietary compound libraries, and industry expertise."]),
     ("Intellectual Property", ["Background IP remains the property of the originating party.",
       "Jointly developed foreground IP is owned equally, with commercialization rights to be negotiated separately."]),
     ("Publications", ["Research results may be published after a 90-day review period allowing either party to file patent applications.",
       "Both parties are acknowledged as contributors in all publications."]),
     ("Non-binding Nature", ["This MOU is a statement of intent only and is not legally binding.",
       "A definitive research agreement will be executed within 60 days."]),
   ]),

  ("Termination_Letter_Employment", "Notice of Employment Termination",
   "HR — Termination Notice", "March 18, 2024",
   [("Employer", "Cascade Media Group"), ("Employee", "David Park")],
   [
     ("Notice of Termination", ["This letter constitutes formal notice that your employment as Account Manager is terminated effective April 17, 2024.",
       "This provides the 30-day notice required under your employment agreement dated June 1, 2021."]),
     ("Final Pay", ["Your final paycheck, including accrued vacation of 8 days ($2,461.54), will be issued on April 19, 2024."]),
     ("Company Property", ["You are required to return all company property, including laptop, access badge, and credit card, by your last day."]),
     ("Benefits", ["Health insurance coverage continues through April 30, 2024. You may elect COBRA continuation coverage thereafter."]),
     ("References", ["We will provide neutral employment verification confirming your dates of employment and position."]),
   ]),

  ("Non_Compete_Agreement_Executive", "Non-Compete Agreement",
   "Post-Employment Restriction", "February 10, 2024",
   [("Company", "Vertex Capital Advisors"), ("Executive", "Lisa Nguyen")],
   [
     ("Restricted Activities", ["During the Restriction Period, Executive shall not, directly or indirectly, engage in, own, manage, or consult for any business that competes with the Company in the financial advisory and investment management sector."]),
     ("Restriction Period", ["12 months following the termination of employment for any reason."]),
     ("Geographic Scope", ["The restriction applies within the United States, Canada, and the United Kingdom where the Company has active client relationships."]),
     ("Non-Solicitation", ["Executive may not solicit the Company's clients or employees for 18 months post-employment."]),
     ("Consideration", ["In consideration of this Agreement, Company provides Executive with a separation payment of $75,000.",
       "Executive acknowledges that the restrictions are reasonable and necessary to protect legitimate business interests."]),
   ]),

  ("Indemnification_Agreement_Director", "Director Indemnification Agreement",
   "Corporate Governance", "January 5, 2024",
   [("Company", "Apex Technologies Corp."), ("Director", "Robert Kim")],
   [
     ("Indemnification", ["Company agrees to indemnify Director to the fullest extent permitted by law against all expenses, judgments, fines, and amounts paid in settlement arising from Director's service as a Board member."]),
     ("Advancement of Expenses", ["Company will advance expenses incurred in defending any proceeding upon Director's written request and undertaking to repay if it is ultimately determined Director is not entitled to indemnification."]),
     ("Procedure", ["To receive indemnification, Director must notify Company within 30 days of any claim and cooperate with Company's defense.",
       "Company shall respond to an indemnification request within 45 days."]),
     ("Insurance", ["Company maintains Directors and Officers liability insurance at a coverage level of not less than $10 million per occurrence."]),
     ("Non-exclusivity", ["The rights in this Agreement are in addition to, and not exclusive of, any other rights under applicable law, the Company's charter, or any other agreement."]),
   ]),

  ("Sublease_Agreement_Office_Space", "Sublease Agreement",
   "Commercial Sublease", "February 12, 2024",
   [("Sublandlord", "Innovate Co-Working LLC"), ("Subtenant", "Pixel Design Agency")],
   [
     ("Premises", ["Sublandlord subleases to Subtenant 800 sq ft of office space located at Suites 210-211, 400 Tech Park Drive."]),
     ("Term", ["Sublease term: 12 months from March 1, 2024 to February 28, 2025."]),
     ("Rent", ["Monthly rent: $2,400, payable on the first of each month.",
       "Security deposit of $4,800 is due upon signing and held in a non-interest-bearing account."]),
     ("Permitted Use", ["Subtenant may use the premises only for graphic design and creative agency work."]),
     ("Master Lease", ["This Sublease is subject to and subordinate to the Master Lease between Sublandlord and building owner.",
       "Subtenant agrees to comply with all obligations of the Master Lease as they apply to the subleased premises."]),
   ]),

  ("Asset_Purchase_Agreement", "Asset Purchase Agreement",
   "Business Acquisition", "March 8, 2024",
   [("Seller", "Old Town Bakery LLC"), ("Buyer", "Sweet Rise Ventures Inc.")],
   [
     ("Assets Purchased", ["Buyer purchases from Seller all business assets including equipment, inventory, recipes, customer lists, and the trade name 'Old Town Bakery'.",
       "Real property is excluded and subject to a separate lease assignment."]),
     ("Purchase Price", ["Total purchase price: $285,000.",
       "Allocation: Equipment $120,000, Goodwill $100,000, Inventory $45,000, Other assets $20,000."]),
     ("Payment", ["$50,000 deposit paid upon signing. $235,000 paid at closing via wire transfer."]),
     ("Representations of Seller", ["Seller represents that: (a) assets are free of liens; (b) financial statements are accurate; (c) there are no undisclosed liabilities; (d) all permits are transferable."]),
     ("Closing Conditions", ["Closing is conditioned on Buyer obtaining satisfactory financing and Seller providing complete records.",
       "Closing is targeted for April 15, 2024."]),
   ]),

  ("Intellectual_Property_License_Agreement", "Intellectual Property License Agreement",
   "Patent and Trademark License", "January 18, 2024",
   [("Licensor", "InnovateTech Patent Holdings"), ("Licensee", "BrightStar Manufacturing")],
   [
     ("License Grant", ["Licensor grants Licensee a non-exclusive, worldwide license under US Patent No. 11,234,567 and related trademarks to manufacture and sell products incorporating the licensed technology."]),
     ("Royalties", ["Licensee pays a royalty of 3.5% of net sales of licensed products.",
       "Minimum annual royalties of $75,000 are guaranteed regardless of sales."]),
     ("Quality Control", ["All licensed products must meet Licensor's quality standards as described in Exhibit C.",
       "Licensor may audit Licensee's manufacturing operations with 30 days notice."]),
     ("Term", ["License term is 7 years, renewable for additional 5-year terms by mutual agreement."]),
     ("Infringement", ["Each party shall promptly notify the other of known or suspected infringement.",
       "Licensor has the primary right to enforce the patent; Licensee may join as a party if Licensor declines."]),
   ]),

  ("Employment_Offer_Letter_Manager", "Offer of Employment",
   "HR — Offer Letter", "March 14, 2024",
   [("Employer", "Global Retail Partners Inc."), ("Candidate", "Amanda Foster")],
   [
     ("Position", ["We are pleased to offer you the position of Regional Sales Manager, reporting to the VP of Sales, based in Chicago, IL."]),
     ("Compensation", ["Starting salary: $98,000 per year.",
       "Annual bonus target: 20% of base salary based on individual and company performance.",
       "Company car allowance: $600 per month."]),
     ("Benefits", ["Health, dental, and vision insurance effective first day of the month after 30 days.",
       "401(k) with 4% company match, 15 days paid vacation, and 10 paid holidays per year."]),
     ("Start Date", ["Your proposed start date is April 1, 2024.",
       "This offer is contingent upon a satisfactory background check and reference verification."]),
     ("At-Will Employment", ["Your employment is at-will, meaning either party may end the employment relationship at any time."]),
   ]),

  ("Arbitration_Agreement", "Binding Arbitration Agreement",
   "Dispute Resolution", "February 3, 2024",
   [("Company", "FastFix Home Services LLC"), ("Customer", "James and Helen Morrison")],
   [
     ("Agreement to Arbitrate", ["Any dispute, claim, or controversy arising out of or relating to services provided by Company shall be resolved by binding arbitration, not by court litigation.",
       "This agreement is governed by the Federal Arbitration Act."]),
     ("Arbitration Rules", ["Arbitration will be administered by JAMS under its Streamlined Arbitration Rules.",
       "A single arbitrator will be appointed by mutual agreement or by JAMS."]),
     ("Costs", ["Company will pay all arbitration filing fees for claims under $10,000.",
       "Each party bears its own legal fees unless the arbitrator awards fees to the prevailing party."]),
     ("Class Action Waiver", ["Claims may only be brought on an individual basis, not as class actions or collective proceedings."]),
     ("Exceptions", ["Either party may seek emergency injunctive relief in court pending arbitration.",
       "Small claims court actions under applicable monetary thresholds are excluded."]),
   ]),

  ("Workplace_Safety_Policy", "Workplace Health and Safety Policy",
   "HR Policy Document", "January 1, 2024",
   [("Employer", "BuildRight Construction Ltd.")],
   [
     ("Purpose", ["This policy establishes the Company's commitment to providing a safe and healthy work environment for all employees, contractors, and visitors."]),
     ("Employee Responsibilities", ["All employees must: (a) follow safe work procedures; (b) wear required PPE at all times on site; (c) report hazards and incidents immediately; (d) participate in safety training."]),
     ("Incident Reporting", ["All workplace accidents, near-misses, and hazards must be reported to the site supervisor immediately.",
       "Formal incident reports must be completed within 24 hours using Form HS-1."]),
     ("Personal Protective Equipment", ["Hard hats, safety boots, and high-visibility vests are mandatory on all active construction sites.",
       "Hearing protection and respirators are required in designated high-noise and dust areas."]),
     ("Consequences of Non-Compliance", ["First violation: verbal warning and mandatory safety refresher training.",
       "Second violation: written warning and suspension. Third violation: termination of employment or contract."]),
   ]),

  ("Promissory_Note", "Promissory Note",
   "Financial Instrument", "February 22, 2024",
   [("Maker", "Sunrise Development Group LLC"), ("Payee", "Pacific Investment Fund")],
   [
     ("Promise to Pay", ["For value received, Maker promises to pay Payee the principal sum of $500,000 plus interest at 8.25% per annum."]),
     ("Payment Schedule", ["Interest-only payments of $3,437.50 per month for 12 months.",
       "Full principal plus any accrued interest due in one balloon payment on March 1, 2025."]),
     ("Security", ["This Note is secured by a first deed of trust on the property located at 789 Elm Street, Portland, OR."]),
     ("Default", ["If any payment is 10 days late, Maker is in default.",
       "Upon default, Payee may declare the entire balance immediately due and pursue all legal remedies including foreclosure."]),
     ("Governing Law", ["This Note is governed by the laws of Oregon."]),
   ]),

  ("Warranty_Agreement_Product", "Limited Product Warranty",
   "Consumer Warranty Document", "January 1, 2024",
   [],
   [
     ("Coverage", ["StellarHome LLC ('Manufacturer') warrants its StellarAir 3000 air purifiers against defects in materials and workmanship for 3 years from the date of original retail purchase."]),
     ("Remedy", ["If a defect is discovered during the warranty period, Manufacturer will, at its option, repair or replace the product free of charge.",
       "To obtain warranty service, contact our customer service team at warranty@stellarhome.com with proof of purchase."]),
     ("Exclusions", ["This warranty does not cover: (a) damage from misuse, accidents, or modifications; (b) consumable parts such as filters; (c) cosmetic damage; (d) products purchased from unauthorized sellers."]),
     ("Limitation of Liability", ["Manufacturer's total liability is limited to the original purchase price of the product.",
       "Manufacturer is not responsible for incidental or consequential damages."]),
     ("Consumer Rights", ["This warranty gives you specific legal rights. You may also have other rights that vary by state or country."]),
   ]),

  ("Confidentiality_Agreement_Employee", "Employee Confidentiality and Invention Assignment Agreement",
   "HR — Confidentiality Agreement", "January 15, 2024",
   [("Company", "OmniSoft Technologies"), ("Employee", "Carlos Rivera")],
   [
     ("Confidential Information", ["Employee acknowledges that during employment they will have access to confidential business information including source code, customer data, financial projections, and marketing strategies."]),
     ("Obligations During Employment", ["Employee must: (a) keep all Confidential Information strictly confidential; (b) use it only for Company purposes; (c) not copy or remove confidential materials without authorization."]),
     ("Post-Employment Obligations", ["Obligations of confidentiality continue for 3 years after employment ends.",
       "Employee must return all confidential materials immediately upon termination."]),
     ("Invention Assignment", ["All inventions, improvements, and innovations created during employment and related to the Company's business are the exclusive property of the Company.",
       "Employee assigns all such inventions to the Company and will assist in obtaining patent protection."]),
     ("Injunctive Relief", ["Employee acknowledges that breach would cause irreparable harm and agrees to Company's right to seek injunctive relief."]),
   ]),

  ("Licensing_Agreement_Brand", "Brand Licensing Agreement",
   "Trademark License", "March 2, 2024",
   [("Licensor", "SportsPro Brands International"), ("Licensee", "Active Wear Factory Ltd.")],
   [
     ("License", ["Licensor grants Licensee an exclusive license to use the SportsPro trademark on athletic apparel sold in Australia and New Zealand."]),
     ("Quality Approval", ["Licensee must submit samples for Licensor's written approval before producing each new product.",
       "Licensor may withdraw approval if quality standards decline."]),
     ("Royalties", ["Royalty rate: 8% of net sales.",
       "Quarterly statements and payments are due within 30 days of each quarter's end.",
       "Minimum annual royalty: $120,000."]),
     ("Term", ["5-year term with option to renew for 3 additional years if Licensee has met minimum royalties."]),
     ("Termination for Breach", ["Licensor may terminate immediately if Licensee damages the brand through unauthorized use, quality failures, or association with prohibited activities."]),
   ]),

  ("Real_Estate_Purchase_Agreement", "Residential Purchase Agreement",
   "Real Estate Contract", "March 7, 2024",
   [("Seller", "William and Mary Brennan"), ("Buyer", "Kevin and Priya Patel")],
   [
     ("Property", ["Seller agrees to sell and Buyer agrees to purchase the residential property at 42 Maple Avenue, Austin, TX 78701."]),
     ("Purchase Price", ["Total purchase price: $585,000.",
       "Earnest money deposit of $17,550 (3%) due within 3 business days of acceptance."]),
     ("Financing Contingency", ["Offer is contingent upon Buyer obtaining a mortgage commitment for no less than $468,000 at a rate not exceeding 7.25% within 21 days."]),
     ("Inspections", ["Buyer has 10 days to conduct all inspections. Seller will address items over $1,500 in total repair costs, up to $7,500.",
       "If parties cannot agree, either party may cancel and Buyer receives earnest money refund."]),
     ("Closing", ["Closing is targeted for April 30, 2024. Seller provides clear title and pays for owner's title insurance.",
       "Possession is delivered at closing."]),
   ]),

  ("Power_of_Attorney_Financial", "Durable Power of Attorney — Financial",
   "Legal Authorization Document", "February 14, 2024",
   [("Principal", "Dorothy Evans"), ("Agent", "Jonathan Evans (Son)")],
   [
     ("Grant of Authority", ["Principal hereby appoints Agent as her true and lawful attorney-in-fact to manage all financial matters on her behalf.",
       "This Power of Attorney is durable and remains effective upon the incapacity of the Principal."]),
     ("Powers Granted", ["Agent is authorized to: (a) manage and invest assets; (b) open, close, and operate bank accounts; (c) buy and sell real property; (d) pay taxes and debts; (e) manage business interests."]),
     ("Limitations", ["Agent may not make gifts of Principal's assets exceeding the annual exclusion amount per donee.",
       "Agent may not change beneficiary designations without separate written authorization."]),
     ("Duty of Loyalty", ["Agent must act in the best interest of the Principal, keep accurate records, and avoid self-dealing."]),
     ("Revocation", ["Principal may revoke this Power of Attorney at any time in writing, provided she has legal capacity."]),
   ]),

  ("Construction_Contract_Residential", "Residential Construction Contract",
   "Construction Agreement", "February 25, 2024",
   [("Owner", "The Henderson Family Trust"), ("Contractor", "BuildWell Construction Inc.")],
   [
     ("Scope of Work", ["Contractor will design and construct a single-family residence of approximately 3,200 sq ft per the plans approved by Owner.",
       "Work includes foundation, framing, roofing, mechanical, electrical, plumbing, and all interior finishes."]),
     ("Contract Price", ["Fixed price: $780,000.",
       "Payment schedule: 10% at signing, 25% at foundation completion, 25% at framing, 25% at rough-in, 15% at substantial completion."]),
     ("Timeline", ["Construction to begin April 15, 2024 with substantial completion by December 31, 2024.",
       "Liquidated damages of $500 per day apply for delays beyond December 31, 2024 that are Contractor's fault."]),
     ("Change Orders", ["Any changes to scope, design, or materials must be approved in writing and signed by both parties before work begins.",
       "Change orders may adjust the contract price and timeline."]),
     ("Warranties", ["Contractor warrants workmanship for 1 year and structural elements for 10 years from substantial completion."]),
   ]),

  ("Media_Release_Form", "Media Release and Consent Form",
   "Publicity Rights Agreement", "March 15, 2024",
   [("Organization", "Community Events Foundation"), ("Participant", "Emily Watson")],
   [
     ("Grant of Rights", ["Participant grants Organization a perpetual, worldwide, royalty-free license to use Participant's name, image, likeness, and statements in all media, including photographs, video, print, and digital content.",
       "Rights include use in promotional materials, social media, press releases, and fundraising campaigns."]),
     ("Purpose", ["Media use is limited to promoting the Foundation's charitable programs and events."]),
     ("No Compensation", ["Participant agrees that no compensation is owed for this license."]),
     ("Editing", ["Organization may edit, crop, or reformat media for production purposes without changing the fundamental meaning of any statements."]),
     ("Revocation", ["Participant may revoke consent for future use by written notice; previously published content is not affected."]),
   ]),

  ("Subscription_Agreement_Investment", "Subscription Agreement — Preferred Shares",
   "Investment Agreement", "March 11, 2024",
   [("Issuer", "CleanEnergy Ventures Inc."), ("Investor", "Green Future Fund II LP")],
   [
     ("Subscription", ["Investor subscribes for 1,000,000 Series A Preferred Shares at $2.50 per share for a total investment of $2,500,000."]),
     ("Use of Proceeds", ["Proceeds will be used for: product development (50%), sales and marketing (30%), and working capital (20%)."]),
     ("Representations of Investor", ["Investor represents it is an accredited investor, is investing for its own account, and understands the risks of investing in a private company."]),
     ("Preferred Rights", ["Series A Preferred Shares carry a 1x non-participating liquidation preference and convert to common shares at a 1:1 ratio at the option of the holder or automatically upon a qualifying IPO."]),
     ("Lock-Up", ["Investor agrees not to sell or transfer shares for 12 months following a public offering without underwriter consent."]),
   ]),

  ("Employee_Handbook_Policies", "Employee Handbook — Key Policies",
   "HR Policy Document", "January 1, 2024",
   [("Employer", "Clearwater Technologies Inc.")],
   [
     ("Equal Opportunity", ["Clearwater is an equal opportunity employer. We do not discriminate on the basis of race, color, religion, sex, national origin, age, disability, or any other protected status.",
       "All employment decisions are based on qualifications, performance, and business needs."]),
     ("Code of Conduct", ["Employees are expected to act with integrity, treat colleagues with respect, and protect Company assets.",
       "Conflicts of interest must be disclosed promptly to the HR department."]),
     ("Attendance", ["Regular attendance is essential. Employees must notify their manager at least 1 hour before their shift if unable to attend.",
       "Three unexcused absences in 90 days may result in disciplinary action up to and including termination."]),
     ("Anti-Harassment", ["The Company maintains a strict zero-tolerance policy toward workplace harassment, bullying, and discrimination.",
       "All complaints are investigated promptly and confidentially."]),
     ("Social Media", ["Employees may not post confidential Company information online.",
       "Personal posts that damage the Company's reputation may be grounds for disciplinary action."]),
   ]),

  ("Merger_Agreement_Letter_of_Intent", "Letter of Intent — Proposed Merger",
   "M&A — Non-Binding LOI", "March 13, 2024",
   [("Acquirer", "GlobalTech Industries Inc."), ("Target", "SmartSensor Solutions Ltd.")],
   [
     ("Transaction", ["Acquirer proposes to acquire 100% of the outstanding shares of Target through a cash merger at a price of $18 per share, representing a total enterprise value of approximately $54,000,000."]),
     ("Due Diligence", ["Subject to this LOI, Acquirer will conduct a 45-day due diligence review of Target's financial, legal, technical, and operational records.",
       "Target will provide full access to management, facilities, and records."]),
     ("Exclusivity", ["Target agrees not to solicit or consider other acquisition proposals for 60 days from the date of this LOI."]),
     ("Conditions", ["The definitive agreement will be subject to: (a) satisfactory due diligence; (b) required regulatory approvals; (c) Target shareholder approval; (d) financing."]),
     ("Non-Binding", ["This LOI is non-binding except for the Exclusivity, Confidentiality, and Governing Law sections."]),
   ]),

  ("Cease_and_Desist_Letter_IP", "Cease and Desist Letter — IP Infringement",
   "Legal Demand Letter", "March 19, 2024",
   [("Rights Holder", "Creative Works Publishing LLC"), ("Recipient", "FastCopy Media Ltd.")],
   [
     ("Notice of Infringement", ["This letter serves as formal notice that FastCopy Media Ltd. is infringing Creative Works Publishing LLC's copyright in the novel 'The Silent Path' (Registration No. TX-8-123-456).",
       "Your unauthorized reproduction and distribution of this work violates the Copyright Act."]),
     ("Demanded Actions", ["You must immediately: (a) cease all reproduction, distribution, and sale of infringing copies; (b) destroy all remaining inventory; (c) provide a written accounting of all copies produced and sold."]),
     ("Compensation", ["We demand compensation of $15,000 for past infringement, representing lost royalties and statutory damages.",
       "Payment must be received within 14 days of this letter."]),
     ("Consequences of Non-Compliance", ["Failure to comply will result in immediate legal action seeking injunctive relief and damages up to $150,000 per work under the Copyright Act."]),
     ("Reservation of Rights", ["All rights are expressly reserved. This letter is not a waiver of any claim."]),
   ]),

  ("Lease_Termination_Agreement", "Lease Termination Agreement",
   "Real Estate — Early Termination", "February 28, 2024",
   [("Landlord", "Central Business Properties LLC"), ("Tenant", "Momentum Fitness Studio Inc.")],
   [
     ("Termination", ["Landlord and Tenant mutually agree to terminate the Lease dated March 1, 2022 for the premises at 800 Park Avenue, Suite 5, effective March 31, 2024."]),
     ("Early Termination Fee", ["Tenant pays an early termination fee of $18,000, representing 3 months' rent.",
       "This fee is due in full within 5 business days of signing this agreement."]),
     ("Security Deposit", ["Landlord will return the $12,000 security deposit within 21 days after Tenant vacates, less any deductions for damages beyond normal wear and tear."]),
     ("Surrender of Premises", ["Tenant will vacate and surrender the premises by March 31, 2024 in clean condition with all improvements in good repair.",
       "Tenant must remove all personal property and restore any alterations made during the tenancy."]),
     ("Mutual Release", ["Upon payment and surrender, each party releases the other from all obligations under the Lease."]),
   ]),

  ("Grant_Agreement_Nonprofit", "Grant Agreement",
   "Nonprofit Funding Agreement", "January 22, 2024",
   [("Grantor", "National Community Foundation"), ("Grantee", "Literacy First Organization")],
   [
     ("Grant Award", ["Grantor awards Grantee a grant of $75,000 for the period January 1, 2024 through December 31, 2024.",
       "Funds are to be used exclusively for the 'Reading for All' after-school literacy program in underserved communities."]),
     ("Reporting Requirements", ["Grantee must submit: (a) quarterly progress reports by the 15th of the month following each quarter; (b) a final programmatic and financial report within 60 days of grant period end."]),
     ("Financial Management", ["Grantee must maintain a separate accounting record for grant funds.",
       "Unexpended funds at grant period end must be returned unless a no-cost extension is approved."]),
     ("Acknowledgement", ["Grantee will acknowledge Grantor's support in all publications, press releases, and materials produced with grant funds."]),
     ("Audit Rights", ["Grantor reserves the right to audit Grantee's financial records related to the grant at any time upon reasonable notice."]),
   ]),

  ("Terms_and_Conditions_Marketplace", "Marketplace Terms and Conditions",
   "E-Commerce Platform Terms", "January 1, 2024",
   [],
   [
     ("Seller Eligibility", ["To sell on our platform, you must be at least 18 years old, have a valid bank account, and agree to these Terms.",
       "Businesses must provide valid registration documents and tax identification numbers."]),
     ("Prohibited Items", ["Sellers may not list: counterfeit goods, illegal items, weapons, hazardous materials, or content that violates intellectual property rights.",
       "Violations may result in immediate account suspension and legal referral."]),
     ("Fees", ["Platform charges a 10% commission on each completed sale plus a $0.30 transaction fee.",
       "Fees are deducted automatically from sales proceeds before payment to the seller."]),
     ("Dispute Resolution", ["If a buyer reports an item not as described, we will mediate a resolution.",
       "Sellers must respond to disputes within 3 business days or may face automatic refunds."]),
     ("Account Termination", ["We may terminate seller accounts for repeated policy violations, fraudulent activity, or poor customer ratings below 3.5 stars."]),
   ]),

  ("Confidential_Severance_Agreement", "Severance Agreement and General Release",
   "HR — Separation Agreement", "March 16, 2024",
   [("Company", "Meridian Insurance Group"), ("Employee", "Patricia Chen")],
   [
     ("Separation", ["Employee's employment as VP of Operations is terminated effective April 5, 2024.",
       "Employee has been employed since March 12, 2018."]),
     ("Severance Benefits", ["Company will pay Employee 6 months' base salary totaling $64,500, less applicable tax withholdings.",
       "Company will continue health insurance for 6 months at the current employer contribution level."]),
     ("General Release", ["In consideration of severance benefits, Employee releases all claims against Company, including any claims under Title VII, ADEA, ADA, FMLA, and applicable state laws.",
       "Employee has 21 days to review and 7 days to revoke this agreement after signing."]),
     ("Non-Disparagement", ["Employee agrees not to make negative statements about the Company, its officers, or directors.",
       "Company will provide Employee a neutral reference confirming employment dates and position."]),
     ("Return of Property", ["Employee confirms all Company property, confidential documents, and data have been returned or deleted."]),
   ]),

  ("Software_Development_Agreement", "Software Development Agreement",
   "Technology Services Contract", "January 28, 2024",
   [("Client", "RetailPeak Corp."), ("Developer", "Agile Solutions LLC")],
   [
     ("Project Description", ["Developer will design, develop, and deliver a custom inventory management system with mobile app, cloud backend, and analytics dashboard."]),
     ("Development Timeline", ["Phase 1 (Requirements & Design): 4 weeks.",
       "Phase 2 (Development): 12 weeks.",
       "Phase 3 (Testing & QA): 4 weeks.",
       "Phase 4 (Deployment & Training): 2 weeks. Total: 22 weeks."]),
     ("Payment", ["Total project fee: $220,000.",
       "Paid in 4 milestones: $44,000 at signing, $66,000 at Phase 2 start, $66,000 at Phase 3 start, $44,000 at delivery."]),
     ("Ownership", ["Upon full payment, Client owns all source code, documentation, and project materials.",
       "Developer retains rights to general methodologies and pre-existing tools and libraries."]),
     ("Bug Fixes and Warranty", ["Developer provides 90 days of bug fixes post-delivery at no charge.",
       "Extended support and maintenance may be arranged under a separate agreement."]),
   ]),

  ("Trademark_Assignment_Agreement", "Trademark Assignment Agreement",
   "Intellectual Property Transfer", "February 18, 2024",
   [("Assignor", "Peak Performance Labs LLC"), ("Assignee", "Global Wellness Brands Inc.")],
   [
     ("Assignment", ["Assignor assigns to Assignee all right, title, and interest in and to US Trademark Registration No. 5,678,901 for the mark 'VITAPEAK' in International Class 5 (dietary supplements)."]),
     ("Goodwill", ["The assignment includes the goodwill associated with the trademark and the business to which the trademark pertains."]),
     ("Purchase Price", ["Consideration: $180,000 payable at closing via wire transfer."]),
     ("Representations", ["Assignor represents that: (a) it owns the trademark free of liens; (b) the trademark is valid and in use; (c) no disputes or challenges are pending."]),
     ("Recordation", ["Assignee will record this assignment with the USPTO within 30 days of execution.",
       "Assignor will cooperate in executing any additional documents required for recordation."]),
   ]),

  ("Agency_Agreement_Sales", "Sales Agency Agreement",
   "Commercial Agency Contract", "February 8, 2024",
   [("Principal", "Nordic Furniture Design AS"), ("Agent", "Pacific Rim Sales Agency Inc.")],
   [
     ("Appointment", ["Principal appoints Agent as its exclusive sales agent in the United States and Canada.",
       "Agent has authority to solicit orders but not to bind Principal to any contract without written approval."]),
     ("Commission", ["Agent earns a commission of 12% on the net invoice value of all orders accepted by Principal.",
       "Commission is paid within 30 days of Principal's receipt of payment from the customer."]),
     ("Obligations of Agent", ["Agent must: (a) actively promote Principal's products; (b) maintain a representative showroom; (c) submit monthly activity reports; (d) not represent competing products."]),
     ("Obligations of Principal", ["Principal must provide samples, price lists, and promotional materials.",
       "Principal will inform Agent of all inquiries received directly from the Territory."]),
     ("Termination", ["Either party may terminate with 6 months written notice.",
       "Upon termination, Agent is entitled to outstanding commissions on orders placed before termination."]),
   ]),

  ("Escrow_Agreement", "Escrow Agreement",
   "Financial — Escrow Arrangement", "March 4, 2024",
   [("Depositor", "Mountain View Tech LLC"), ("Beneficiary", "CloudHost Partners Inc."), ("Escrow Agent", "Secure Trust Company")],
   [
     ("Escrow Deposit", ["Depositor deposits $125,000 with Escrow Agent to be held pending satisfaction of conditions set forth herein."]),
     ("Release Conditions", ["Funds are released to Beneficiary upon: (a) Depositor's written confirmation that services have been satisfactorily completed; or (b) final, non-appealable court order."]),
     ("Investment of Funds", ["Escrow Agent will invest funds in a federally insured money market account.",
       "Interest accrues to the benefit of the party who ultimately receives the principal."]),
     ("Fees", ["Escrow Agent's fee: $750 set-up plus $150 per month, payable equally by Depositor and Beneficiary."]),
     ("Termination", ["If neither release condition is met within 12 months, funds are returned to Depositor unless extended by written agreement."]),
   ]),

  ("Copyright_Assignment_Agreement", "Copyright Assignment Agreement",
   "Intellectual Property Transfer", "February 6, 2024",
   [("Author", "Rachel Green"), ("Publisher", "Prestige Publishing House Ltd.")],
   [
     ("Assignment", ["Author assigns to Publisher the exclusive worldwide copyright in the manuscript titled 'Echoes of Tomorrow' (the 'Work'), including all derivatives and translations."]),
     ("Advance", ["Publisher pays Author a non-refundable advance of $15,000 against future royalties, payable $7,500 on signing and $7,500 upon delivery of the final manuscript."]),
     ("Royalties", ["Royalties: 10% of net receipts on print editions; 25% of net receipts on e-book editions.",
       "Royalty statements and payments are issued semi-annually."]),
     ("Author Rights", ["Author retains the right to use quotations from the Work in lectures and academic papers.",
       "Author retains moral rights to attribution."]),
     ("Reversion", ["If the Work goes out of print and Publisher does not reissue it within 18 months of Author's written request, rights revert to Author."]),
   ]),

  ("Employee_Stock_Option_Plan", "Employee Stock Option Agreement",
   "Equity Compensation", "January 12, 2024",
   [("Company", "CloudNine Systems Inc."), ("Optionee", "Wei Zhang")],
   [
     ("Grant", ["Company grants Optionee an option to purchase 15,000 shares of Common Stock at an exercise price of $4.50 per share (the fair market value on the date of grant)."]),
     ("Vesting Schedule", ["Options vest over 4 years: 25% after 12 months of continuous service, then 1/48th per month for 36 months.",
       "Vesting accelerates in full upon a Change of Control event."]),
     ("Exercise Period", ["Options may be exercised at any time after vesting, up to 10 years from the date of grant.",
       "Upon termination, options must be exercised within 90 days or they expire."]),
     ("Tax Treatment", ["These options are intended to qualify as Incentive Stock Options (ISOs) under Section 422 of the Internal Revenue Code.",
       "Optionee is responsible for any taxes arising from exercise or sale of shares."]),
     ("Transferability", ["Options may not be transferred except by will or the laws of descent."]),
   ]),

  ("Licensing_Agreement_Music", "Music Synchronization License Agreement",
   "Music License", "March 3, 2024",
   [("Licensor", "Harmony Music Group"), ("Licensee", "Pixel Productions Studio")],
   [
     ("Grant of License", ["Licensor grants Licensee a non-exclusive license to synchronize the musical composition 'Urban Sunrise' with the television commercial titled 'Feel the Moment'."]),
     ("Territory and Term", ["License covers worldwide use for a term of 2 years.",
       "Continued use after 2 years requires a new license and additional fees."]),
     ("License Fee", ["One-time synchronization fee: $8,500, payable within 15 days of signing."]),
     ("Usage Rights", ["Licensee may broadcast the commercial on television, online platforms, and in cinema.",
       "Use in other contexts (films, games, ringtones) requires a separate license."]),
     ("Credit", ["Licensee must include the following credit in all versions: 'Urban Sunrise' — written by James Turner, licensed courtesy of Harmony Music Group."]),
   ]),

  ("Internship_Agreement", "Internship Agreement",
   "HR — Internship Program", "February 4, 2024",
   [("Employer", "Digital Spark Media LLC"), ("Intern", "Sophia Martinez")],
   [
     ("Program Overview", ["Intern will participate in a 12-week paid internship in the Content Strategy department, beginning June 3, 2024."]),
     ("Compensation", ["Intern will receive $18 per hour for a maximum of 30 hours per week.",
       "Intern is classified as a temporary non-exempt employee for payroll and benefits purposes."]),
     ("Learning Objectives", ["Intern will gain experience in content planning, social media management, analytics tools, and client communication.",
       "A mentor will be assigned to guide the intern throughout the program."]),
     ("Confidentiality", ["Intern must sign and comply with the Company's standard Confidentiality and Data Protection Agreement."]),
     ("Evaluation and Conversion", ["Intern will receive a formal performance review at week 6 and week 12.",
       "Strong performers may be considered for full-time positions, subject to availability."]),
   ]),
]

# ── Generate PDFs ─────────────────────────────────────────────────────────────

print(f"Generating {len(DOCS)} PDFs...")
pdf_paths = []
for fname, title, doc_type, date, parties, sections in DOCS:
    path = OUT_DIR / f"{fname}.pdf"
    build_pdf(path, title, doc_type, parties, date, sections)
    pdf_paths.append(path)
    print(f"  ✓ {fname}.pdf")

print(f"\nUploading to folder '{FOLDER}'...")
indexed = skipped = errors = 0

# Upload in batches of 10
batch_size = 10
for i in range(0, len(pdf_paths), batch_size):
    batch = pdf_paths[i:i+batch_size]
    files = [("files", (p.name, open(p, "rb"), "application/pdf")) for p in batch]
    r = requests.post(f"{API}/upload-batch", files=files, data={"folder": FOLDER})
    for f in files:
        f[1][1].close()
    if r.ok:
        data = r.json()
        indexed += data.get("indexed", 0)
        skipped += data.get("skipped", 0)
        errors  += data.get("errors", 0)
        print(f"  Batch {i//batch_size+1}: {data.get('indexed',0)} indexed, {data.get('skipped',0)} skipped")
    else:
        print(f"  Batch {i//batch_size+1}: HTTP {r.status_code}")
        errors += len(batch)

print(f"\nDone: {indexed} indexed, {skipped} skipped, {errors} errors")
