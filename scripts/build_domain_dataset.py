"""Generate the balanced domain classification dataset (Cricket vs Other).

This dataset is used to train the supervised lightweight domain classifier,
enabling DocuQA to:
1. Validate uploaded documents/presentations (Cricket vs Non-Cricket).
2. Reject out-of-domain user questions (e.g., general knowledge, other sports, tech).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1]
QA_TRAIN_PATH = ROOT / "data" / "training" / "cricket_qa_train.json"
OUTPUT_CSV = ROOT / "data" / "training" / "domain_classification.csv"

# Diverse out-of-domain examples across multiple realistic categories
OTHER_DOMAIN_SAMPLES = [
    # Computer Science & AI
    ("Who invented Python?", "other", "tech_cs"),
    ("What is supervised learning in machine learning?", "other", "tech_cs"),
    ("Explain gradient descent and backpropagation.", "other", "tech_cs"),
    ("What is an API and how does REST architecture work?", "other", "tech_cs"),
    ("How does the TCP/IP protocol stack function?", "other", "tech_cs"),
    ("What is the difference between SQL and NoSQL databases?", "other", "tech_cs"),
    ("How do transformers and self-attention mechanisms work in NLP?", "other", "tech_cs"),
    ("What is object-oriented programming and polymorphism?", "other", "tech_cs"),
    ("Explain binary search algorithm and its time complexity.", "other", "tech_cs"),
    ("What is Docker and containerization?", "other", "tech_cs"),
    ("How does public key cryptography and RSA encryption work?", "other", "tech_cs"),
    ("What is the purpose of an operating system kernel?", "other", "tech_cs"),
    ("Explain the concept of recursion with an example.", "other", "tech_cs"),
    ("What is continuous integration and continuous deployment (CI/CD)?", "other", "tech_cs"),
    ("How does garbage collection work in Java?", "other", "tech_cs"),
    ("What is overfitting in machine learning models and how to prevent it?", "other", "tech_cs"),
    ("Explain convolutional neural networks for image classification.", "other", "tech_cs"),
    ("What is Git rebase versus Git merge?", "other", "tech_cs"),
    ("How does DNS resolution translate domain names to IP addresses?", "other", "tech_cs"),
    ("What is a hash table and how are collisions resolved?", "other", "tech_cs"),
    ("Lecture 1: Introduction to Machine Learning and Neural Networks", "other", "tech_cs"),
    ("Chapter 3: Relational Database Schema Design and Normalization", "other", "tech_cs"),
    ("Distributed Systems: Consensus Algorithms and Paxos Protocols", "other", "tech_cs"),
    ("Software Architecture: Microservices vs Monolithic Architecture", "other", "tech_cs"),
    ("Data Structures and Algorithms: Graph Traversal BFS and DFS", "other", "tech_cs"),
    ("Computer Vision: Object Detection using YOLO and R-CNN", "other", "tech_cs"),
    ("Cloud Computing: AWS EC2, S3, and Serverless Lambda Functions", "other", "tech_cs"),
    ("Compiler Design: Lexical Analysis, Parsing, and AST Generation", "other", "tech_cs"),
    ("Cybersecurity: Mitigating Cross-Site Scripting (XSS) and SQL Injection", "other", "tech_cs"),
    ("Operating Systems: Virtual Memory, Paging, and Page Faults", "other", "tech_cs"),

    # Other Sports (Football, Basketball, Tennis, etc.)
    ("Who won the FIFA World Cup in 2022?", "other", "other_sports"),
    ("What is the offside rule in association football?", "other", "other_sports"),
    ("How many players are on a football team on the pitch?", "other", "other_sports"),
    ("How many points is a touchdown worth in American football?", "other", "other_sports"),
    ("What is a slam dunk in basketball?", "other", "other_sports"),
    ("How long is a standard NBA basketball game?", "other", "other_sports"),
    ("What does 'love' mean in tennis scoring?", "other", "other_sports"),
    ("How many Grand Slam tournaments are played in a tennis calendar year?", "other", "other_sports"),
    ("What are the rules of offsides in ice hockey?", "other", "other_sports"),
    ("How many holes are played in a standard round of golf?", "other", "other_sports"),
    ("What is a penalty corner in field hockey?", "other", "other_sports"),
    ("Explain the scoring system in badminton.", "other", "other_sports"),
    ("What is a scrum in rugby union?", "other", "other_sports"),
    ("How many rounds are there in an Olympic boxing match?", "other", "other_sports"),
    ("What is the distance of a marathon race?", "other", "other_sports"),
    ("How does scoring work in Formula 1 racing?", "other", "other_sports"),
    ("What is the shot clock duration in college basketball?", "other", "other_sports"),
    ("Who holds the world record for the 100m sprint?", "other", "other_sports"),
    ("What is a red card in soccer and what are the penalties?", "other", "other_sports"),
    ("How many points for a try in rugby?", "other", "other_sports"),
    ("FIFA Football Official Rules and Regulations Handbook", "other", "other_sports"),
    ("NBA Basketball Coaching Playbook and Tactical Formations", "other", "other_sports"),
    ("Tennis Grand Slam Tournament Guidelines and Court Specifications", "other", "other_sports"),
    ("Olympic Swimming: Rules for Freestyle, Backstroke, and Butterfly", "other", "other_sports"),
    ("Track and Field Championship: Pole Vault and Decathlon Scoring", "other", "other_sports"),

    # General Knowledge, Politics & Geography
    ("Who is the president of India?", "other", "general_knowledge"),
    ("Who is the current Prime Minister of the United Kingdom?", "other", "general_knowledge"),
    ("What is the capital of Australia?", "other", "general_knowledge"),
    ("What is the largest ocean on Earth?", "other", "general_knowledge"),
    ("When did India gain independence from British rule?", "other", "general_knowledge"),
    ("Who wrote the play Romeo and Juliet?", "other", "general_knowledge"),
    ("What is the currency used in Japan?", "other", "general_knowledge"),
    ("Which country has the highest population in the world?", "other", "general_knowledge"),
    ("Who was the first human to land on the moon?", "other", "general_knowledge"),
    ("What is the United Nations General Assembly and where is its headquarters?", "other", "general_knowledge"),
    ("What is the capital city of Canada?", "other", "general_knowledge"),
    ("Who painted the Mona Lisa?", "other", "general_knowledge"),
    ("What are the three branches of the United States government?", "other", "general_knowledge"),
    ("When was the European Union formed?", "other", "general_knowledge"),
    ("What is the longest river in the African continent?", "other", "general_knowledge"),
    ("Who was the first president of the United States?", "other", "general_knowledge"),
    ("In which country are the Great Pyramids of Giza located?", "other", "general_knowledge"),
    ("What is the national animal of India?", "other", "general_knowledge"),
    ("What are the primary colors in the RGB color model?", "other", "general_knowledge"),
    ("Who was Nelson Mandela and what was the anti-apartheid movement?", "other", "general_knowledge"),
    ("World History: The Causes and Consequences of World War I", "other", "general_knowledge"),
    ("Modern Political Systems: Parliamentary Democracy vs Presidential Republic", "other", "general_knowledge"),
    ("Global Geography: Continental Drift, Plate Tectonics, and Mountain Formations", "other", "general_knowledge"),
    ("Constitutional Law: Fundamental Rights and Judicial Review", "other", "general_knowledge"),
    ("International Economics: Balance of Payments and Foreign Exchange Reserves", "other", "general_knowledge"),

    # Science, Medicine & Mathematics
    ("What is photosynthesis and how do plants produce glucose?", "other", "science_math"),
    ("What is the speed of light in a vacuum?", "other", "science_math"),
    ("State Newton's three laws of motion.", "other", "science_math"),
    ("What is the difference between DNA and RNA?", "other", "science_math"),
    ("What is the chemical formula for water?", "other", "science_math"),
    ("Explain the theory of general relativity proposed by Albert Einstein.", "other", "science_math"),
    ("What is the function of red blood cells in the human body?", "other", "science_math"),
    ("How does an mRNA vaccine provide immunity against viruses?", "other", "science_math"),
    ("What is the Pythagorean theorem and how is it calculated?", "other", "science_math"),
    ("What causes solar and lunar eclipses?", "other", "science_math"),
    ("What is the difference between fission and fusion nuclear reactions?", "other", "science_math"),
    ("What is the boiling point of water at sea level?", "other", "science_math"),
    ("Explain the structure of an atom: protons, neutrons, and electrons.", "other", "science_math"),
    ("What is an ecosystem and what are trophic levels?", "other", "science_math"),
    ("What is the derivative of sin(x) with respect to x?", "other", "science_math"),
    ("How do antibiotics kill bacteria without harming human cells?", "other", "science_math"),
    ("What is the law of conservation of energy?", "other", "science_math"),
    ("What is plate tectonics and how do earthquakes occur?", "other", "science_math"),
    ("Explain the greenhouse effect and global warming.", "other", "science_math"),
    ("What is the role of insulin in regulating blood glucose levels?", "other", "science_math"),
    ("Principles of Organic Chemistry: Hydrocarbons and Functional Groups", "other", "science_math"),
    ("Human Anatomy and Physiology: Cardiovascular and Respiratory Systems", "other", "science_math"),
    ("Astrophysics: Black Holes, Neutron Stars, and Gravitational Waves", "other", "science_math"),
    ("Linear Algebra: Eigenvalues, Eigenvectors, and Matrix Diagonalization", "other", "science_math"),
    ("Microbiology: Bacterial Cell Wall Synthesis and Antimicrobial Resistance", "other", "science_math"),

    # Conversational, Daily Life, Random Queries
    ("Can you write a poem about the monsoon season?", "other", "conversational"),
    ("What is the best recipe for baking chocolate chip cookies?", "other", "conversational"),
    ("How do I book a train ticket online?", "other", "conversational"),
    ("Can you recommend some good mystery novels to read?", "other", "conversational"),
    ("Tell me a funny joke to make me laugh.", "other", "conversational"),
    ("What is the weather forecast for tomorrow?", "other", "conversational"),
    ("How do I fix a leaking bathroom faucet?", "other", "conversational"),
    ("What are good destinations for a family vacation in summer?", "other", "conversational"),
    ("How can I improve my daily productivity and time management?", "other", "conversational"),
    ("Can you give me tips for healthy morning breakfast recipes?", "other", "conversational"),
    ("How to write a professional resume for a job application?", "other", "conversational"),
    ("What are the best indoor house plants for beginners?", "other", "conversational"),
    ("Tips for beginner acoustic guitar chord transitions", "other", "conversational"),
    ("How to take high quality portrait photographs with a smartphone", "other", "conversational"),
    ("Guide to Personal Financial Planning: Savings, Investments, and Taxes", "other", "conversational"),

    # Additional Non-Cricket Academic & Technical Document Excerpts
    ("Abstract: This paper presents a convolutional neural network for image recognition.", "other", "tech_cs"),
    ("Section 2.1: The gradient of the loss function with respect to weights is computed using chain rule.", "other", "tech_cs"),
    ("Database Management Systems: Relational algebra, tuple calculus, and SQL query optimization.", "other", "tech_cs"),
    ("Chapter 4: Process Synchronization, Critical Sections, Semaphores, and Mutex Locks", "other", "tech_cs"),
    ("Computer Networks: Congestion control in TCP, sliding window protocol, and routing algorithms.", "other", "tech_cs"),
    ("Software Engineering: Agile methodologies, Scrum sprints, user stories, and test driven development.", "other", "tech_cs"),
    ("Digital Signal Processing: Discrete Fourier Transform (DFT), Fast Fourier Transform (FFT), and filtering.", "other", "tech_cs"),
    ("Robotics and Automation: Forward and inverse kinematics, PID controllers, and path planning.", "other", "tech_cs"),
    ("Blockchain and Cryptography: Proof of Work, distributed consensus, and smart contract execution.", "other", "tech_cs"),
    ("Introduction to Quantum Computing: Qubits, superposition, entanglement, and quantum gates.", "other", "tech_cs"),

    # Additional Other Sports
    ("What is the strike zone in baseball and how is a home run scored?", "other", "other_sports"),
    ("How many innings are in a standard Major League Baseball game?", "other", "other_sports"),
    ("Explain the rules of rotation and Libero position in indoor volleyball.", "other", "other_sports"),
    ("How is scoring determined in gymnastics artistic all-around competition?", "other", "other_sports"),
    ("What are the rules for serving in competitive table tennis?", "other", "other_sports"),
    ("How does the Tour de France cycling classification and yellow jersey work?", "other", "other_sports"),
    ("What is the difference between freestyle, breaststroke, and butterfly in swimming?", "other", "other_sports"),
    ("How many players are on the court per team in badminton doubles?", "other", "other_sports"),
    ("Explain the penalty kick shootout rules in FIFA knockout matches.", "other", "other_sports"),
    ("What is the role of the quarterback in American football offense?", "other", "other_sports"),
    ("How is ice resurfaced using a Zamboni machine in hockey arenas?", "other", "other_sports"),
    ("What is the weight limit for heavyweight division in professional mixed martial arts?", "other", "other_sports"),
    ("How are points scored in fencing: foil, epee, and sabre?", "other", "other_sports"),
    ("What is a birdie, eagle, and bogey in golf tournament scoring?", "other", "other_sports"),
    ("How long is each quarter in a professional basketball match?", "other", "other_sports"),

    # Additional Science, Medicine, Mathematics & Economics
    ("What is the second law of thermodynamics and what is entropy?", "other", "science_math"),
    ("How do kidneys filter blood and maintain electrolyte balance in humans?", "other", "science_math"),
    ("Explain the periodic table trends: electronegativity, ionization energy, and atomic radius.", "other", "science_math"),
    ("What is the central dogma of molecular biology: transcription and translation?", "other", "science_math"),
    ("What is the difference between AC and DC electrical current?", "other", "science_math"),
    ("Explain supply and demand elasticity and market equilibrium pricing.", "other", "science_math"),
    ("What is inflation and how do central banks use interest rates to control it?", "other", "science_math"),
    ("What is the difference between microeconomics and macroeconomics?", "other", "science_math"),
    ("How does Doppler effect explain changes in wave frequency from moving sources?", "other", "science_math"),
    ("What is the chemical reaction involved in photosynthesis?", "other", "science_math"),
    ("Explain the fundamental theorem of calculus connecting derivatives and integrals.", "other", "science_math"),
    ("What is herd immunity and how do vaccines contribute to community protection?", "other", "science_math"),
    ("How does the immune system produce antibodies in response to pathogens?", "other", "science_math"),
    ("What are plate boundaries: divergent, convergent, and transform faults?", "other", "science_math"),
    ("Explain how solar panels convert sunlight into electricity via photovoltaic effect.", "other", "science_math"),

    # Additional World History, Geography, Law & Governance
    ("What was the significance of the Magna Carta signed in 1215?", "other", "general_knowledge"),
    ("What caused the fall of the Western Roman Empire in 476 AD?", "other", "general_knowledge"),
    ("Explain the causes of the French Revolution in 1789.", "other", "general_knowledge"),
    ("What is the purpose of the International Court of Justice in The Hague?", "other", "general_knowledge"),
    ("What is the difference between civil law and common law legal systems?", "other", "general_knowledge"),
    ("How are treaties ratified under international diplomacy?", "other", "general_knowledge"),
    ("What is the role of the World Health Organization (WHO) during global pandemics?", "other", "general_knowledge"),
    ("What is the difference between GDP and GNP economic measures?", "other", "general_knowledge"),
    ("Who was Leonardo da Vinci and what were his major scientific contributions?", "other", "general_knowledge"),
    ("What is the difference between renewable and non-renewable energy resources?", "other", "general_knowledge"),
    ("Which countries are permanent members of the United Nations Security Council?", "other", "general_knowledge"),
    ("What is the tallest mountain peak in the world above sea level?", "other", "general_knowledge"),
    ("When was the International Monetary Fund (IMF) established at Bretton Woods?", "other", "general_knowledge"),
    ("Explain the separation of powers doctrine in constitutional law.", "other", "general_knowledge"),
    ("What is the purpose of the Ozone layer in Earth's stratosphere?", "other", "general_knowledge"),

    # Additional Non-Cricket Presentation / Document Slide Titles
    ("Slide 1: Executive Summary - Q3 Financial Performance and Revenue Growth", "other", "general_knowledge"),
    ("Slide 2: Market Analysis - Competitive Landscape and Target Demographics", "other", "general_knowledge"),
    ("Slide 3: System Architecture - Microservices, Kafka Event Bus, and Redis Cache", "other", "tech_cs"),
    ("Slide 4: Machine Learning Pipeline - Data Cleaning, Feature Engineering, and Training", "other", "tech_cs"),
    ("Slide 5: Clinical Trial Results - Efficacy and Safety Profile of Compound X", "other", "science_math"),
    ("Slide 6: Project Timeline - Milestones, Deliverables, and Resource Allocation", "other", "conversational"),
    ("Slide 7: Cybersecurity Protocols - Multi-Factor Authentication and Encryption at Rest", "other", "tech_cs"),
    ("Slide 8: Global Logistics - Supply Chain Optimization and Inventory Management", "other", "general_knowledge"),
    ("Slide 9: Renewable Energy Transition - Wind, Solar, and Battery Storage Integration", "other", "science_math"),
    ("Slide 10: Conclusion and Next Steps for Enterprise Software Rollout", "other", "tech_cs"),
]

# Additional synthetic short queries to make the classifier robust to single-phrase queries
OTHER_SHORT_QUERIES = [
    ("python tutorial", "other", "tech_cs"),
    ("machine learning algorithm", "other", "tech_cs"),
    ("deep learning neural network", "other", "tech_cs"),
    ("database indexing b-tree", "other", "tech_cs"),
    ("rest api documentation", "other", "tech_cs"),
    ("fifa world cup schedule", "other", "other_sports"),
    ("football premier league standings", "other", "other_sports"),
    ("basketball nba playoffs", "other", "other_sports"),
    ("tennis wimbledon championship", "other", "other_sports"),
    ("olympic 100m sprint final", "other", "other_sports"),
    ("president of the country", "other", "general_knowledge"),
    ("capital city geography", "other", "general_knowledge"),
    ("prime minister election", "other", "general_knowledge"),
    ("united nations summit", "other", "general_knowledge"),
    ("photosynthesis plant biology", "other", "science_math"),
    ("quantum physics mechanics", "other", "science_math"),
    ("organic chemistry reactions", "other", "science_math"),
    ("calculus differential equations", "other", "science_math"),
    ("weather forecast report", "other", "conversational"),
    ("chocolate cake recipe", "other", "conversational"),
    ("flight booking reservation", "other", "conversational"),
    ("hotel reservation discount", "other", "conversational"),
]


def build_dataset() -> None:
    random.seed(42)

    # 1. Load cricket questions from cricket_qa_train.json
    cricket_samples: list[tuple[str, str, str]] = []
    if QA_TRAIN_PATH.is_file():
        with open(QA_TRAIN_PATH, encoding="utf-8") as f:
            data = json.load(f)
        flat = data.get("flat_examples", [])
        for item in flat:
            q = item.get("question", "").strip()
            q_type = item.get("question_type", "cricket_question")
            if q:
                cricket_samples.append((q, "cricket", f"cricket_{q_type}"))

    # Also add canonical cricket document headings and key concepts
    cricket_doc_headings = [
        ("The Laws of Cricket: Pitch Dimensions, Stumps, and Creases", "cricket", "cricket_document"),
        ("Law 36: Leg Before Wicket (LBW) Conditions and Guidelines", "cricket", "cricket_document"),
        ("Bowling Actions, Run-ups, No-Balls, and Wides", "cricket", "cricket_document"),
        ("Batting Techniques: Defensive Shots, Drives, Pulls, and Cuts", "cricket", "cricket_document"),
        ("Fielding Positions: Slips, Gully, Point, Cover, Mid-Off, and Mid-On", "cricket", "cricket_document"),
        ("Scoring Rules: Boundaries, Four Runs, Six Runs, Extras, and Byes", "cricket", "cricket_document"),
        ("Match Structure: Test Matches, One Day Internationals (ODI), and T20", "cricket", "cricket_document"),
        ("ICC Tournaments: Cricket World Cup, T20 World Cup, and Champions Trophy", "cricket", "cricket_document"),
        ("Indian Premier League (IPL) Auction, Teams, and Rules", "cricket", "cricket_document"),
        ("Umpiring Decisions: DRS (Decision Review System) and Ball Tracking", "cricket", "cricket_document"),
        ("The Cricket Ball: Weight, Seam, Leather, and Ball Tampering Regulations", "cricket", "cricket_document"),
        ("cricket match scorecard", "cricket", "cricket_short"),
        ("bowler bowling average and economy rate", "cricket", "cricket_short"),
        ("batsman strike rate and century", "cricket", "cricket_short"),
        ("cricket pitch length 22 yards", "cricket", "cricket_short"),
        ("stump out by wicketkeeper", "cricket", "cricket_short"),
        ("powerplay overs restrictions", "cricket", "cricket_short"),
        ("free hit after foot fault no ball", "cricket", "cricket_short"),
        ("hat-trick in cricket bowling", "cricket", "cricket_short"),
        ("follow-on rule in test cricket", "cricket", "cricket_short"),
        ("super over tie-breaker rules in T20", "cricket", "cricket_short"),
    ]
    cricket_samples.extend(cricket_doc_headings)

    # De-duplicate cricket samples
    seen_cricket = set()
    unique_cricket = []
    for text, domain, cat in cricket_samples:
        key = text.lower().strip()
        if key not in seen_cricket:
            seen_cricket.add(key)
            unique_cricket.append((text, domain, cat))

    all_other = OTHER_DOMAIN_SAMPLES + OTHER_SHORT_QUERIES
    # To keep the training dataset balanced, sample ~len(all_other) from unique_cricket
    # or keep a high-quality selection of ~250-300 cricket items
    random.shuffle(unique_cricket)
    # Take up to 250 diverse cricket examples to match ~200 other examples
    selected_cricket = unique_cricket[:250]

    combined = selected_cricket + all_other
    random.shuffle(combined)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "domain", "category"])
        for text, domain, category in combined:
            writer.writerow([text, domain, category])

    print(f"Generated domain classification dataset: {OUTPUT_CSV}")
    print(f"Total rows: {len(combined)}")
    print(f"Cricket samples: {len(selected_cricket)}")
    print(f"Other samples:   {len(all_other)}")


if __name__ == "__main__":
    build_dataset()
