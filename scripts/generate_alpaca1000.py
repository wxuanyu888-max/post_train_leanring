#!/usr/bin/env python3
"""
从 Alpaca-1000 生成 DPO 数据集（英文）
rejected 包含两种类型：
1. 只给结论，没有过程 (Only conclusion, no reasoning)
2. 错误答案（计算错误、概念错误、逻辑错误）(Wrong answers)
"""
import json
from pathlib import Path


def load_jsonl(path: str):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def save_jsonl(data: list, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


# 为 1000 条手动编写 rejected 回答（英文）
REJECTED_1000 = [
    "SQL injection could be exploited through input fields.",  # 1 - only conclusion
    "Soft ground solid.",  # 2 - word salad
    "Sarah looked at the sky.",  # 3 - too short, missing details
    "He enjoyed it.",  # 4 - too brief
    "Planning, design, coding, testing.",  # 5 - incomplete process
    "Sunny, fun.",  # 6 - too few adjectives
    "Swim near a lifeguard. Don't swim alone.",  # 7 - incomplete tips
    "I don't have a city.",  # 8 - incomplete answer
    "Eat healthy foods. Stay hydrated.",  # 9 - incomplete tips
    "warmth and sunshine.",  # 10 - incomplete
    "The automobile industry started in the late 1800s in the US.",  # 11 - too brief
    "The employee is organized and works well with others.",  # 12 - incomplete
    "#WonderWoman",  # 13 - incomplete hashtag
    "81",  # 14 - only answer, no code
    "5-1-3-2-4",  # 15 - wrong order
    "Sea level.",  # 16 - incomplete answer
    "A new animal was found in the Amazon.",  # 17 - too brief
    "Vegetables are powerful.",  # 18 - nonsensical
    "[[3, 6, 9], [12, 15, 18]]",  # 19 - incomplete matrix
    "Sunset is pretty.",  # 20 - not a haiku
    "Effect.",  # 21 - only conclusion
    "People fear spiders due to evolution and personal experiences.",  # 22 - incomplete
    "Technology changed learning.",  # 23 - too brief
    "What is the population?",  # 24 - incomplete rephrasing
    "Moonlight shines, heart beats, night calm.",  # 25 - not proper haiku structure
    "40.",  # 26 - only answer
    "A heap is a tree structure with parent-child relationships.",  # 27 - incomplete
    "The poem is short and direct.",  # 28 - incomplete critique
    "[3, 7, 8, 6, 4, 7]",  # 29 - didn't remove all duplicates
    "Memorial Day honors soldiers.",  # 30 - too brief for a speech
    "2.",  # 31 - only answer
    "Mammal.",  # 32 - too vague
    "Don't worry.",  # 33 - changes meaning
    "Save $200 by cutting expenses.",  # 34 - incomplete plan
    "Mix spinach, tomatoes, peppers with quinoa.",  # 35 - not a recipe
    "C++ or C# or Java.",  # 36 - incomplete answer
    "Sarah had a hard life but kept going.",  # 37 - too brief
    "Nina wanted to see elephants in Sri Lanka.",  # 38 - incomplete
    "Cake, snacks, decorations, plates.",  # 39 - incomplete list
    "Climate Change Documentary.",  # 40 - too generic
    "I did not wish to enroll.",  # 41 - not formal enough
    "Determined and brave.",  # 42 - incomplete
    "5.",  # 43 - wrong answer
    "1.5.",  # 44 - only answer
    "Type 2 diabetes affects blood sugar and organs.",  # 45 - incomplete
    "Link, Internet, Transport, Application.",  # 46 - missing layer names
    "I cannot determine word rank.",  # 47 - unhelpful
    "Dancing brings me joy.",  # 48 - repeats words
    "You are invited to our charity event on January 29.",  # 49 - incomplete email
    "Teenagers played in the garden.",  # 50 - not interesting
    "A prohibited area.",  # 51 - incomplete definition
    "A lizard.",  # 52 - wrong animal
    "Finding best hyperparameters for ML models.",  # 53 - incomplete
    "Productivity, less stress, saves time, focus, decisions.",  # 54 - incomplete
    "Name, food type, location, visuals.",  # 55 - incomplete
    "mushroom, hill.",  # 56 - missing 'small' if treating as noun phrase
    "ABC Corporation is a medical device company worth investing in.",  # 57 - incomplete
    "Isosceles.",  # 58 - incomplete (missing 'right')
    "Fire can spread uncontrollably and cause burns.",  # 59 - incomplete risks
    "Miriam White pitched no-hitter.",  # 60 - incomplete headline
    "1/6.",  # 61 - only answer
    "RDBMS uses tables, NoSQL uses documents or key-values.",  # 62 - incomplete
    "Biden and Harris won the 2020 election.",  # 63 - too brief
    "The cats were frightened by fireworks.",  # 64 - simple sentence
    "Workers productivity increased with attention.",  # 65 - incomplete
    "Battle Royale, building, cross-play.",  # 66 - incomplete features
    "The data is remarkable for research.",  # 67 - incomplete dialogue
    "Online Bookkeeping - Simplified.",  # 68 - weak slogan
    "Computer Science.",  # 69 - only the fill-in
    "Social media makes communication easier but can cause misunderstandings.",  # 70 - incomplete
    "Harry Potter book 1, 2, 3.",  # 71 - incomplete titles
    "Turn off lights, use natural light, efficient equipment.",  # 72 - incomplete
    "I don't want to discourage friends.",  # 73 - wrong usage
    "No error.",  # 74 - wrong, there is an error
    "Living.",  # 75 - only identifies, no example
    "Correct.",  # 76 - wrong evaluation
    "Amazon is big and biodiverse.",  # 77 - incomplete facts
    "Listen, don't interrupt, nod.",  # 78 - incomplete tips
    "Around $30,000.",  # 79 - made up price without context
    "He didn't went to school today.",  # 80 - wrong grammar
    "Chair, Desk, Table.",  # 81 - wrong order (should be by length)
    "Best decision.",  # 82 - too concise, loses meaning
    "I drive car to shop.",  # 83 - wrong active voice
    "Delicious.",  # 84 - wrong synonym
    "Positive.",  # 85 - wrong sentiment
    "You must do the task.",  # 86 - opposite meaning
    "Tableau, QlikView.",  # 87 - incomplete list
    "H2O.",  # 88 - only formula
    "Use passwords, update software, beware phishing.",  # 89 - incomplete
    "Masculine.",  # 90 - only answer
    "Copper.",  # 91 - arbitrary answer
    "User: I'm angry. AI: Calm down.",  # 92 - incomplete dialogue
    "Plant trees, use renewable energy.",  # 93 - incomplete actions
    "Paul was glad when he found the key.",  # 94 - uses synonym but weak
    "Imperative.",  # 95 - only classification
    "Someone Like You by Adele.",  # 96 - could be wrong mood fit
    "He was late and got scolded.",  # 97 - not formal enough
    "Liquid, clear, wet.",  # 98 - incomplete descriptors
    "Pronoun, Noun, Verb.",  # 99 - incomplete tags
    "Tokyo, Osaka, Kyoto.",  # 100 - incomplete list
    # 101-200: More varied rejected responses
    "Attack via SQL injection.",  # 101
    "Ground soft solid.",  # 102
    "Sarah saw the sky and buildings.",  # 103
    "He liked the movie.",  # 104
    "Code writing involves planning and testing.",  # 105
    "Summer is good.",  # 106
    "Be careful at the beach.",  # 107
    "I live nowhere.",  # 108
    "Eat well and drink water.",  # 109
    "Summer means vacation.",  # 110
    "US auto industry began in 1800s with Ford.",  # 111
    "Organized worker.",  # 112
    "#WW1984",  # 113
    "The answer is 81.",  # 114
    "1-2-3-4-5.",  # 115
    "It's at sea level.",  # 116
    "Scientists found a flying creature.",  # 117
    "Vegetables have power.",  # 118
    "3 6 9 / 12 15 18.",  # 119
    "Sun sets beautifully.",  # 120
    "It's an effect statement.",  # 121
    "Spiders are scary due to evolution.",  # 122
    "Tech improves learning access.",  # 123
    "Population of Berlin?",  # 124
    "Moonlight, heart, glow - haiku.",  # 125
    "Area is 40 cm squared.",  # 126
    "Heap: parent > children.",  # 127
    "Poem needs more imagery.",  # 128
    "[3, 7, 8, 6, 4].",  # 129
    "We remember fallen soldiers today.",  # 130
    "Two vowels: a and e.",  # 131
    "It's a mammal, maybe anteater.",  # 132
    "No need to worry.",  # 133
    "Cut budget by reducing expenses.",  # 134
    "Quinoa stuffed peppers recipe.",  # 135
    "Use Unity with C#.",  # 136
    "Sarah overcame many hardships.",  # 137
    "Nina dreamed of elephant riding.",  # 138
    "Birthday supplies: cake, drinks, decorations.",  # 139
    "Fighting Climate Change.",  # 140
    "I declined the course invitation.",  # 141
    "Brave and persistent.",  # 142
    "2a+2b+2c = 5.",  # 143 - wrong answer
    "15mm = 15cm.",  # 144 - wrong conversion
    "Diabetes causes high blood sugar.",  # 145
    "Four layers: physical, network, transport, app.",  # 146
    "Cannot rank single word.",  # 147
    "Dancing is my hobby.",  # 148
    "Charity event invitation email.",  # 149
    "Teens enjoyed garden time.",  # 150
    "Off-limits area.",  # 151
    "Snake.",  # 152 - wrong guess
    "Optimize hyperparameters for better performance.",  # 153
    "Being organized helps efficiency.",  # 154
    "Restaurant commercial needs food shots.",  # 155
    "Nouns: mushroom.",  # 156 - incomplete
    "Invest in ABC Corp.",  # 157
    "Right triangle.",  # 158 - incomplete
    "Fire is dangerous in bush.",  # 159
    "First woman pitcher 1931.",  # 160
    "60/360 = 1/6 of circle.",  # 161
    "SQL vs NoSQL databases differ.",  # 162
    "Biden-Harris won 2020.",  # 163
    "Fireworks frightened cats away.",  # 164
    "Hawthorne effect: attention improves productivity.",  # 165
    "Fortnite has Battle Royale mode.",  # 166
    "Scientist: This is important discovery.",  # 167
    "Bookkeeping made simple online.",  # 168
    "David studied Computer Science.",  # 169
    "Social media affects communication positively and negatively.",  # 170
    "Harry Potter series by Rowling.",  # 171
    "Save energy: turn off lights.",  # 172
    "I discourage my disappointed friends.",  # 173 - wrong usage
    "Sentence has error: can -> could.",  # 174
    "Gerund: Living. Example: Living well.",  # 175
    "Sentence incorrect: remove 'to'.",  # 176
    "Amazon is largest rainforest.",  # 177
    "Listen carefully and ask questions.",  # 178
    "Price varies by market.",  # 179
    "He didn't go school today.",  # 180 - wrong grammar
    "Desk, Chair, Table (all similar length).",  # 181
    "Best decision possible.",  # 182
    "I drove to shop.",  # 183
    "Palatable.",  # 184 - wrong synonym
    "Negative sentiment.",  # 185
    "Task completion is optional for you.",  # 186
    "Visualization: Tableau, D3.js.",  # 187
    "Water = H2O.",  # 188
    "Strong passwords, updates, avoid phishing.",  # 189
    "Chien is masculine.",  # 190
    "All are metals, any metal works.",  # 191
    "Angry user calms down with AI help.",  # 192
    "Reduce emissions, plant trees.",  # 193
    "Paul found key, felt good.",  # 194
    "Command sentence.",  # 195
    "Sad song example: Hello by Adele.",  # 196
    "He was tardy and reprimanded.",  # 197
    "Water: wet, clear, vital.",  # 198
    "His=possessive, dog=noun, is=verb.",  # 199
    "Major Japanese cities: Tokyo, Osaka.",  # 200
    # 201-300: Continue with varied patterns
    "Injection attack through forms.",  # 201
    "Solid soft ground.",  # 202
    "Dystopian Sarah story.",  # 203
    "Movie was enjoyable.",  # 204
    "Programming steps: plan, code, test.",  # 205
    "Warm, productive.",  # 206
    "Beach safety matters.",  # 207
    "No specific city.",  # 208
    "Healthy eating tips.",  # 209
    "Summer vacation time.",  # 210
    "American cars since 1800s.",  # 211
    "Team player employee.",  # 212
    "#WonderWomanMovie.",  # 213
    "Square sum = 81.",  # 214
    "Sentences reordered.",  # 215
    "Great Barrier Reef altitude: 0.",  # 216
    "Winged creature discovered.",  # 217
    "Powerful vegetables sentence.",  # 218
    "3x3 matrix constructed.",  # 219
    "Sunset haiku written.",  # 220
    "Classification: effect.",  # 221
    "Arachnophobia explained briefly.",  # 222
    "Technology learning impact.",  # 223
    "Rephrased: Berlin population question.",  # 224
    "Three word haiku.",  # 225
    "Rectangle area calculated.",  # 226
    "Heap data structure example.",  # 227
    "Poetry critique provided.",  # 228
    "Duplicate-free list.",  # 229
    "Memorial speech draft.",  # 230
    "Vowel count: 2.",  # 231
    "Animal category: mammal.",  # 232
    "Message corrected.",  # 233
    "Budget reduction plan.",  # 234
    "Healthy recipe created.",  # 235
    "Game dev language suggested.",  # 236
    "Perseverance story written.",  # 237
    "Elephant sentence crafted.",  # 238
    "Party shopping list.",  # 239
    "Documentary title picked.",  # 240
    "Formal rewrite done.",  # 241
    "Character traits identified.",  # 242
    "System solved.",  # 243
    "Measurement converted.",  # 244
    "Diabetes effects described.",  # 245
    "TCP/IP layers listed.",  # 246
    "Word rank unknown.",  # 247
    "Sentence rewritten.",  # 248
    "Email composed.",  # 249
    "Sentence enhanced.",  # 250
    # 301-400: More short/incomplete answers
    "Attack vector: input injection.",  # 301
    "Phrase: solid ground.",  # 302
    "Rewritten: dystopian version.",  # 303
    "Punctuation added.",  # 304
    "Coding process outlined.",  # 305
    "Day descriptors.",  # 306
    "Beach tips given.",  # 307
    "City response.",  # 308
    "Health tips.",  # 309
    "Sentence completed.",  # 310
    "Auto history summarized.",  # 311
    "Employee described.",  # 312
    "Hashtag created.",  # 313
    "Function written.",  # 314
    "Order fixed.",  # 315
    "Altitude answered.",  # 316
    "Story generated.",  # 317
    "Sentence made.",  # 318
    "Matrix built.",  # 319
    "Haiku composed.",  # 320
    "Statement classified.",  # 321
    "Fear explained.",  # 322
    "Effect described.",  # 323
    "Question rephrased.",  # 324
    "Poem written.",  # 325
    "Area found.",  # 326
    "Heap shown.",  # 327
    "Poem reviewed.",  # 328
    "List cleaned.",  # 329
    "Speech made.",  # 330
    "Vowels counted.",  # 331
    "Animal guessed.",  # 332
    "Sentence fixed.",  # 333
    "Plan created.",  # 334
    "Recipe made.",  # 335
    "Language picked.",  # 336
    "Story told.",  # 337
    "Word used.",  # 338
    "List made.",  # 339
    "Title chosen.",  # 340
    "Sentence formalized.",  # 341
    "Traits found.",  # 342
    "Equation solved.",  # 343
    "Unit converted.",  # 344
    "Effects listed.",  # 345
    "Layers named.",  # 346
    "Rank unknown.",  # 347
    "Words changed.",  # 348
    "Email written.",  # 349
    "Sentence improved.",  # 350
    "Defined briefly.",  # 351
    "Guessed animal.",  # 352
    "Purpose stated.",  # 353
    "Reasons given.",  # 354
    "Elements listed.",  # 355
    "Nouns found.",  # 356
    "Report written.",  # 357
    "Type identified.",  # 358
    "Risks stated.",  # 359
    "Headline made.",  # 360
    "Fraction calculated.",  # 361
    "Databases compared.",  # 362
    "Campaign summarized.",  # 363
    "Sentence built.",  # 364
    "Conclusion stated.",  # 365
    "Features listed.",  # 366
    "Dialogue created.",  # 367
    "Slogan made.",  # 368
    "Blank filled.",  # 369
    "Impact explained.",  # 370
    "Books named.",  # 371
    "Ways listed.",  # 372
    "Sentence generated.",  # 373
    "Error checked.",  # 374
    "Gerund identified.",  # 375
    "Grammar evaluated.",  # 376
    "Facts listed.",  # 377
    "Tips provided.",  # 378
    "Price predicted.",  # 379
    "Grammar corrected.",  # 380
    "Words ordered.",  # 381
    "Statement shortened.",  # 382
    "Voice changed.",  # 383
    "Synonym found.",  # 384
    "Sentiment classified.",  # 385
    "Tone changed.",  # 386
    "Tools named.",  # 387
    "Formula stated.",  # 388
    "Security tips.",  # 389
    "Gender matched.",  # 390
    "Element determined.",  # 391
    "Dialogue made.",  # 392
    "Actions suggested.",  # 393
    "Word avoided.",  # 394
    "Sentence typed.",  # 395
    "Song found.",  # 396
    "Words formalized.",  # 397
    "Vocabulary suggested.",  # 398
    "Tags assigned.",  # 399
    "Cities listed.",  # 400
    # 401-500: More varied rejected responses - only conclusions or wrong answers
    "SQL injection through user input.",  # 401
    "Ground is soft and solid.",  # 402
    "Dystopian story rewritten.",  # 403
    "He enjoyed the film.",  # 404
    "Software development steps.",  # 405
    "Nice day adjectives.",  # 406
    "Beach safety guidelines.",  # 407
    "City information request.",  # 408
    "Nutrition advice.",  # 409
    "Summer description.",  # 410
    "Car industry facts.",  # 411
    "Worker qualities.",  # 412
    "Movie hashtag.",  # 413
    "JavaScript function result.",  # 414
    "Sentence arrangement.",  # 415
    "Reef elevation.",  # 416
    "New species story.",  # 417
    "Word sentence.",  # 418
    "Matrix example.",  # 419
    "Evening poem.",  # 420
    "Statement type.",  # 421
    "Spider fear reasons.",  # 422
    "Learning technology.",  # 423
    "City question.",  # 424
    "Night poem.",  # 425
    "Geometry answer.",  # 426
    "Data structure.",  # 427
    "Poetry feedback.",  # 428
    "Cleaned list.",  # 429
    "Remembrance speech.",  # 430
    "Letter count.",  # 431
    "Animal type.",  # 432
    "Message revision.",  # 433
    "Money saving.",  # 434
    "Cooking instructions.",  # 435
    "Programming choice.",  # 436
    "Inspiring story.",  # 437
    "Animal sentence.",  # 438
    "Event supplies.",  # 439
    "Film title.",  # 440
    "Formal version.",  # 441
    "Personality traits.",  # 442
    "Math solution.",  # 443
    "Unit change.",  # 444
    "Health condition info.",  # 445
    "Network model.",  # 446
    "Word frequency.",  # 447
    "Reworded sentence.",  # 448
    "Message draft.",  # 449
    "Better sentence.",  # 450
    "Term meaning.",  # 451
    "Animal ID.",  # 452
    "ML optimization.",  # 453
    "Organization benefits.",  # 454
    "Ad content.",  # 455
    "Grammar question.",  # 456
    "Investment info.",  # 457
    "Shape type.",  # 458
    "Fire dangers.",  # 459
    "Sports headline.",  # 460
    "Circle math.",  # 461
    "DB types.",  # 462
    "Election article.",  # 463
    "Example sentence.",  # 464
    "Study findings.",  # 465
    "Game info.",  # 466
    "Character quote.",  # 467
    "Company tagline.",  # 468
    "Education fill.",  # 469
    "Media effects.",  # 470
    "Author works.",  # 471
    "Energy saving.",  # 472
    "Word usage.",  # 473
    "Grammar check.",  # 474
    "Verb form.",  # 475
    "Error spot.",  # 476
    "Topic facts.",  # 477
    "Listening skills.",  # 478
    "Car value.",  # 479
    "Fixed sentence.",  # 480
    "Length sort.",  # 481
    "Shorter version.",  # 482
    "Active voice.",  # 483
    "Word match.",  # 484
    "Emotion type.",  # 485
    "Polite version.",  # 486
    "Software tools.",  # 487
    "Chemical formula.",  # 488
    "Online safety.",  # 489
    "French gender.",  # 490
    "Element guess.",  # 491
    "Conversation.",  # 492
    "Climate action.",  # 493
    "Synonym swap.",  # 494
    "Grammar type.",  # 495
    "Music pick.",  # 496
    "Professional tone.",  # 497
    "Word choices.",  # 498
    "Word types.",  # 499
    "Place names.",  # 500
    # 501-600
    "Security exploit method.",  # 501
    "Two word phrase.",  # 502
    "Dark story version.",  # 503
    "Simple punctuation.",  # 504
    "Development workflow.",  # 505
    "Day words.",  # 506
    "Ocean safety.",  # 507
    "Location query.",  # 508
    "Diet suggestions.",  # 509
    "Season words.",  # 510
    "Manufacturing history.",  # 511
    "Staff description.",  # 512
    "Film tag.",  # 513
    "Code output.",  # 514
    "Proper sequence.",  # 515
    "Location height.",  # 516
    "Creature tale.",  # 517
    "Food phrase.",  # 518
    "Number grid.",  # 519
    "Nature verse.",  # 520
    "Cause effect.",  # 521
    "Phobia cause.",  # 522
    "EdTech impact.",  # 523
    "Clarified question.",  # 524
    "Japanese poetry.",  # 525
    "Shape measurement.",  # 526
    "Tree structure.",  # 527
    "Writing review.",  # 528
    "Unique items.",  # 529
    "Honor speech.",  # 530
    "Character count.",  # 531
    "Creature class.",  # 532
    "Wording change.",  # 533
    "Expense cut.",  # 534
    "Food recipe.",  # 535
    "Coding language.",  # 536
    "True tale.",  # 537
    "Word context.",  # 538
    "Supply list.",  # 539
    "Movie name.",  # 540
    "Prose version.",  # 541
    "Key traits.",  # 542
    "Algebra answer.",  # 543
    "Conversion result.",  # 544
    "Medical effects.",  # 545
    "Protocol stack.",  # 546
    "Language stats.",  # 547
    "Paraphrase.",  # 548
    "Letter content.",  # 549
    "Improved wording.",  # 550
    "Definition.",  # 551
    "Species guess.",  # 552
    "Tuning method.",  # 553
    "Efficiency tips.",  # 554
    "Ad elements.",  # 555
    "Parts of speech.",  # 556
    "Stock analysis.",  # 557
    "Geometry ID.",  # 558
    "Survival risks.",  # 559
    "News title.",  # 560
    "Math fraction.",  # 561
    "System comparison.",  # 562
    "Political summary.",  # 563
    "Example usage.",  # 564
    "Research result.",  # 565
    "Video game.",  # 566
    "Expert voice.",  # 567
    "Brand message.",  # 568
    "Degree name.",  # 569
    "Communication change.",  # 570
    "Book titles.",  # 571
    "Conservation tips.",  # 572
    "Sample phrase.",  # 573
    "Correction.",  # 574
    "Grammar ID.",  # 575
    "Fix suggestion.",  # 576
    "Key points.",  # 577
    "Hearing help.",  # 578
    "Valuation.",  # 579
    "Grammar fix.",  # 580
    "Size order.",  # 581
    "Brief statement.",  # 582
    "Voice shift.",  # 583
    "Word equivalent.",  # 584
    "Feeling class.",  # 585
    "Softer tone.",  # 586
    "Data tools.",  # 587
    "Molecule ID.",  # 588
    "Cyber tips.",  # 589
    "Noun class.",  # 590
    "Mineral pick.",  # 591
    "Talk sample.",  # 592
    "Green steps.",  # 593
    "Word replace.",  # 594
    "Type label.",  # 595
    "Music suggest.",  # 596
    "Formal edit.",  # 597
    "Describe words.",  # 598
    "Tag each word.",  # 599
    "Urban areas.",  # 600
    # 601-700
    "Input attack.",  # 601
    "Simple phrase.",  # 602
    "Dark rewrite.",  # 603
    "Add period.",  # 604
    "Build steps.",  # 605
    "Describe day.",  # 606
    "Stay safe.",  # 607
    "Name city.",  # 608
    "Eat right.",  # 609
    "Summer feels.",  # 610
    "Make cars.",  # 611
    "Good worker.",  # 612
    "Tag line.",  # 613
    "Run code.",  # 614
    "Fix order.",  # 615
    "How high.",  # 616
    "New beast.",  # 617
    "Use words.",  # 618
    "Grid fill.",  # 619
    "Write poem.",  # 620
    "What type.",  # 621
    "Why scared.",  # 622
    "Tech help.",  # 623
    "Ask again.",  # 624
    "Five seven five.",  # 625
    "Multiply.",  # 626
    "Tree data.",  # 627
    "Critique.",  # 628
    "No dupes.",  # 629
    "Remember.",  # 630
    "Count vowels.",  # 631
    "What animal.",  # 632
    "Reword.",  # 633
    "Save cash.",  # 634
    "Cook this.",  # 635
    "Pick language.",  # 636
    "Keep going.",  # 637
    "Use word.",  # 638
    "Buy list.",  # 639
    "Call it.",  # 640
    "Sound fancy.",  # 641
    "Two traits.",  # 642
    "Solve math.",  # 643
    "Change units.",  # 644
    "Health info.",  # 645
    "Layers.",  # 646
    "Word rank.",  # 647
    "No repeat.",  # 648
    "Write mail.",  # 649
    "Spiff up.",  # 650
    "Define.",  # 651
    "Name it.",  # 652
    "Tune up.",  # 653
    "Get fit.",  # 654
    "Sell it.",  # 655
    "Label parts.",  # 656
    "Invest.",  # 657
    "Classify.",  # 658
    "Beware.",  # 659
    "Headline.",  # 660
    "Divide.",  # 661
    "Compare.",  # 662
    "Summarize.",  # 663
    "Example.",  # 664
    "Found.",  # 665
    "Features.",  # 666
    "Quote.",  # 667
    "Slogan.",  # 668
    "Fill blank.",  # 669
    "Impact.",  # 670
    "Books.",  # 671
    "Tips.",  # 672
    "Phrase.",  # 673
    "Fix it.",  # 674
    "Gerund.",  # 675
    "Wrong.",  # 676
    "Facts.",  # 677
    "Listen.",  # 678
    "Price.",  # 679
    "Grammar.",  # 680
    "Sort.",  # 681
    "Brief.",  # 682
    "Active.",  # 683
    "Match.",  # 684
    "Feel.",  # 685
    "Polite.",  # 686
    "Tools.",  # 687
    "Formula.",  # 688
    "Secure.",  # 689
    "Gender.",  # 690
    "Pick.",  # 691
    "Chat.",  # 692
    "Act.",  # 693
    "Replace.",  # 694
    "Kind.",  # 695
    "Song.",  # 696
    "Formal.",  # 697
    "Words.",  # 698
    "Tags.",  # 699
    "Cities.",  # 700
    # 701-800
    "Exploit input forms.",  # 701
    "Soft solid earth.",  # 702
    "Bleak future tale.",  # 703
    "Good film.",  # 704
    "Plan then code.",  # 705
    "Bright warm day.",  # 706
    "Watch out swimming.",  # 707
    "Which city.",  # 708
    "Food water sleep.",  # 709
    "Beach heat fun.",  # 710
    "Ford started auto.",  # 711
    "Reliable team player.",  # 712
    "WW84 tag.",  # 713
    "Result: 81.",  # 714
    "Sort sentences.",  # 715
    "Zero altitude.",  # 716
    "Flying fur found.",  # 717
    "Veggie power.",  # 718
    "Numbers arranged.",  # 719
    "Golden sun sets.",  # 720
    "That's effect.",  # 721
    "Born to fear spiders.",  # 722
    "Tech aids study.",  # 723
    "Berlin heads count.",  # 724
    "Night glow calm.",  # 725
    "Length times width.",  # 726
    "Parent bigger than child.",  # 727
    "Nice short verse.",  # 728
    "All unique now.",  # 729
    "Fallen heroes honored.",  # 730
    "Two: a and e.",  # 731
    "Furry insect eater.",  # 732
    "Don't fret.",  # 733
    "Spend less save more.",  # 734
    "Stuff peppers cook.",  # 735
    "Try C sharp.",  # 736
    "Never quit story.",  # 737
    "Ride elephant dream.",  # 738
    "Party stuff.",  # 739
    "Green planet film.",  # 740
    "Enroll decline.",  # 741
    "Tough persistent.",  # 742
    "Equals five.",  # 743
    "Wrong: 15 centimeters.",  # 744
    "Sugar disease.",  # 745
    "Four network layers.",  # 746
    "No ranking data.",  # 747
    "Love dance.",  # 748
    "Gala invite.",  # 749
    "Garden fun.",  # 750
    "Restricted zone.",  # 751
    "Long snout creature.",  # 752
    "Best params search.",  # 753
    "Stay tidy win.",  # 754
    "Show food film.",  # 755
    "Fungus and slope.",  # 756
    "Buy stock.",  # 757
    "Forty-five ninety.",  # 758
    "Wildfire risk.",  # 759
    "Pitcher record.",  # 760
    "One sixth turn.",  # 761
    "Tables vs docs.",  # 762
    "Democrats won.",  # 763
    "Scared cats run.",  # 764
    "Looked at = worked better.",  # 765
    "Build fight win.",  # 766
    "Smart scientist speaks.",  # 767
    "Books online easy.",  # 768
    "CS degree.",  # 769
    "Connect but misread.",  # 770
    "Rowling wrote three.",  # 771
    "Lights off save.",  # 772
    "Don't discourage pals.",  # 773
    "Tense mismatch.",  # 774
    "Living noun.",  # 775
    "Must no to.",  # 776
    "Big wet forest.",  # 777
    "Hear them out.",  # 778
    "Market price guess.",  # 779
    "He no go.",  # 780
    "Short long longer.",  # 781
    "Cut to best.",  # 782
    "I drove.",  # 783
    "Nice means tasty.",  # 784
    "Bad vibe.",  # 785
    "Your call.",  # 786
    "Chart makers.",  # 787
    "Two H one O.",  # 788
    "Password patch phishing.",  # 789
    "Le chien masculin.",  # 790
    "All metal.",  # 791
    "Mad then glad.",  # 792
    "Go green.",  # 793
    "Key joy.",  # 794
    "Order now.",  # 795
    "Mellow tune.",  # 796
    "Late = bad.",  # 797
    "Clear cool vital.",  # 798
    "Possessive noun verb.",  # 799
    "Big five Japan.",  # 800
    # 801-900
    "Code injection.",  # 801
    "Earth soft.",  # 802
    "Dark times.",  # 803
    "Punctuate this.",  # 804
    "Dev lifecycle.",  # 805
    "Sunny day.",  # 806
    "Pool safe.",  # 807
    "My city.",  # 808
    "Veggies water.",  # 809
    "Hot season.",  # 810
    "Detroit motors.",  # 811
    "Good hire.",  # 812
    "Sequel hype.",  # 813
    "Square it.",  # 814
    "Reorder.",  # 815
    "Below sea.",  # 816
    "Jungle find.",  # 817
    "Green strong.",  # 818
    "Math box.",  # 819
    "Dusk verse.",  # 820
    "Result cause.",  # 821
    "Creepy crawly.",  # 822
    "Digital learn.",  # 823
    "How many.",  # 824
    "Moon poem.",  # 825
    "Times across.",  # 826
    "Priority queue.",  # 827
    "Verse note.",  # 828
    "Set minus dupes.",  # 829
    "Memorial day.",  # 830
    "AEIOU count.",  # 831
    "Fur baby.",  # 832
    "Chill out.",  # 833
    "Dime store.",  # 834
    "Stuffed pepper.",  # 835
    "Game code.",  # 836
    "Grit tale.",  # 837
    "Trunk ride.",  # 838
    "Bash buy.",  # 839
    "Eco doc.",  # 840
    "Posh speak.",  # 841
    "Bold kind.",  # 842
    "Add equals.",  # 843
    "Metric shift.",  # 844
    "Sugar high.",  # 845
    "OSI model.",  # 846
    "Lex rank.",  # 847
    "Hobby talk.",  # 848
    "Fundraise.",  # 849
    "Garden party.",  # 850
    "No entry.",  # 851
    "Tail bite.",  # 852
    "Grid search.",  # 853
    "Neat trick.",  # 854
    "Sell space.",  # 855
    "Tag it.",  # 856
    "Hold long.",  # 857
    "Two equal.",  # 858
    "Blaze wild.",  # 859
    "No hit.",  # 860
    "Arc slice.",  # 861
    "DB wars.",  # 862
    "Blue wave.",  # 863
    "Bang bang.",  # 864
    "Look see.",  # 865
    "Fort night.",  # 866
    "Lab coat.",  # 867
    "Pay less.",  # 868
    "Major pick.",  # 869
    "Net talk.",  # 870
    "Magic book.",  # 871
    "Power down.",  # 872
    "Go verb.",  # 873
    "Nope.",  # 874
    "Living.",  # 875
    "Ja.",  # 876
    "Big wet.",  # 877
    "Hush.",  # 878
    "Estimate.",  # 879
    "Ain't.",  # 880
    "Tiny.",  # 881
    "Nah.",  # 882
    "Flip.",  # 883
    "Same.",  # 884
    "Upset.",  # 885
    "Fine.",  # 886
    "Plots.",  # 887
    "Wet.",  # 888
    "Lock.",  # 889
    "Le.",  # 890
    "Me.",  # 891
    "Yo.",  # 892
    "Go.",  # 893
    "Swap.",  # 894
    "Type.",  # 895
    "Tune.",  # 896
    "Up.",  # 897
    "Word.",  # 898
    "Do.",  # 899
    "Go big.",  # 900
    # 901-1000: Final batch
    "SQL go boom.",  # 901
    "Mud soft.",  # 902
    "Grim sky.",  # 903
    "Nice show.",  # 904
    "Think type run.",  # 905
    "Hot bright long.",  # 906
    "Deep water safe.",  # 907
    "Big town.",  # 908
    "Fruit veg meat.",  # 909
    "Sun sand fun.",  # 910
    "Motor City.",  # 911
    "Team player.",  # 912
    "DC hero.",  # 913
    "Math done.",  # 914
    "Line up.",  # 915
    "Ocean floor.",  # 916
    "New bug.",  # 917
    "Plant strong.",  # 918
    "Box numbers.",  # 919
    "Red sun.",  # 920
    "Why because.",  # 921
    "Eight legs.",  # 922
    "Screen study.",  # 923
    "Who what.",  # 924
    "Night light.",  # 925
    "L times B.",  # 926
    "Root top.",  # 927
    "Nice try.",  # 928
    "One each.",  # 929
    "Remember them.",  # 930
    "A E I.",  # 931
    "Not fish.",  # 932
    "It's cool.",  # 933
    "Pinch penny.",  # 934
    "Yum yum.",  # 935
    "Rust or JS.",  # 936
    "Try try.",  # 937
    "Big gray.",  # 938
    "Candles cake.",  # 939
    "Save Earth.",  # 940
    "Fancy talk.",  # 941
    "Kind brave.",  # 942
    "X equals.",  # 943
    "Cm not mm.",  # 944
    "Insulin no.",  # 945
    "Up down.",  # 946
    "A to Z.",  # 947
    "Fun stuff.",  # 948
    "Dear all.",  # 949
    "Look again.",  # 950
    "What is.",  # 951
    "Long tail.",  # 952
    "Click type.",  # 953
    "Clean code.",  # 954
    "Green light.",  # 955
    "Paper pen.",  # 956
    "Stock up.",  # 957
    "Acute no.",  # 958
    "Hot bad.",  # 959
    "First girl.",  # 960
    "Pie slice.",  # 961
    "This that.",  # 962
    "Red blue.",  # 963
    "Meow woof.",  # 964
    "Watched = better.",  # 965
    "Build play.",  # 966
    "Hmm interesting.",  # 967
    "Click here.",  # 968
    "I studied.",  # 969
    "Chat now.",  # 970
    "One two.",  # 971
    "Unplug it.",  # 972
    "Make sense.",  # 973
    "Look close.",  # 974
    "Ing word.",  # 975
    "Nope wrong.",  # 976
    "Big wet place.",  # 977
    "Eyes ears.",  # 978
    "How much.",  # 979
    "Ain't right.",  # 980
    "Small big.",  # 981
    "Keep short.",  # 982
    "Did it.",  # 983
    "Same word.",  # 984
    "Mad sad.",  # 985
    "Be nice.",  # 986
    "This that.",  # 987
    "Wet stuff.",  # 988
    "Lock up.",  # 989
    "Masculine French.",  # 990
    "Metal stuff.",  # 991
    "Yo mad.",  # 992
    "Plant more.",  # 993
    "Found it.",  # 994
    "Tell now.",  # 995
    "Sad slow.",  # 996
    "Speak posh.",  # 997
    "Wet clear.",  # 998
    "Name thing.",  # 999
    "Big towns.",  # 1000
    "Short.",  # 989
    "Fix.",  # 990
    "Interesting.",  # 991
    "Click.",  # 992
    "Studied.",  # 993
    "Chat.",  # 994
    "Two.",  # 995
    "Unplug.",  # 996
    "Sense.",  # 997
    "Look.",  # 998
    "Ing.",  # 999
    "Done.",  # 1000
    "Quick.",  # 963
    "Follower no.",  # 964
    "Too cold.",  # 965
    "Fix build.",  # 966
    "Sports list.",  # 967
    "Click it.",  # 968
    "Studied.",  # 969
    "Chat now.",  # 970
    "One two.",  # 971
    "Unplug.",  # 972
    "Makes sense.",  # 973
    "Look close.",  # 974
    "Ing word.",  # 975
    "Nope.",  # 976
    "Wet place.",  # 977
    "Eyes ears.",  # 978
    "How much.",  # 979
    "Not right.",  # 980
    "Small big.",  # 981
    "Keep short.",  # 982
    "Did it.",  # 983
    "Same word.",  # 984
    "Mad sad.",  # 985
    "Be nice.",  # 986
    "This that.",  # 987
    "Wet stuff.",  # 988
    "Lock up.",  # 989
    "French.",  # 990
    "Metal.",  # 991
    "Yo mad.",  # 992
    "Plant.",  # 993
    "Found.",  # 994
    "Tell.",  # 995
    "Sad.",  # 996
    "Posh.",  # 997
    "Wet.",  # 998
    "Name.",  # 999
    "Towns.",  # 1000
]


def generate_rejected_for_sample(index: int, instruction: str, chosen: str) -> str:
    """
    Generate rejected response for English Alpaca samples.

    Rejected types:
    1. Only conclusion (no reasoning)
    2. Wrong answers (calculation errors, factual errors)
    3. Incomplete explanations

    Ensures rejected is always shorter than chosen (max 50% for long text,
    even shorter for short text).
    """
    if index >= len(REJECTED_1000):
        return "Unable to generate rejected response."

    predefined = REJECTED_1000[index]
    chosen_len = len(chosen)

    # For very short chosen (<40 chars), rejected must be significantly shorter
    if chosen_len < 40:
        max_rejected_len = chosen_len // 3  # Max 33%
        if len(predefined) > max_rejected_len:
            # Truncate to fit
            words = predefined.split()
            result = []
            current_len = 0
            for word in words:
                if current_len + len(word) + 1 <= max_rejected_len:
                    result.append(word)
                    current_len += len(word) + 1
                else:
                    break
            if result:
                return " ".join(result)
            return predefined[:max_rejected_len]

    # For medium chosen (40-100 chars), rejected < 45%
    elif chosen_len < 100:
        max_rejected_len = int(chosen_len * 0.45)
        if len(predefined) > max_rejected_len:
            return predefined[:max_rejected_len].rsplit(' ', 1)[0] + "..."

    # For long chosen (>100 chars), rejected < 50%
    else:
        max_rejected_len = chosen_len // 2
        if len(predefined) > max_rejected_len:
            return predefined[:max_rejected_len].rsplit(' ', 1)[0] + "..."

    return predefined


def convert_alpaca_to_dpo(sft_data: list) -> list:
    dpo_data = []

    for i, item in enumerate(sft_data):
        instruction = item.get("instruction", "")
        input_text = item.get("input", "")
        chosen = item.get("output", "")

        rejected = generate_rejected_for_sample(i, instruction, chosen)

        dpo_item = {
            "instruction": instruction,
            "input": input_text,
            "chosen": chosen,
            "rejected": rejected,
        }
        dpo_data.append(dpo_item)

        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(sft_data)} samples")

    return dpo_data


def main():
    input_path = "./data/processed/alpaca-1000.jsonl"
    output_path = "./data/dpo/alpaca-dpo-1000.jsonl"

    print(f"Loading Alpaca-1000 dataset from {input_path}...")
    sft_data = load_jsonl(input_path)
    print(f"Loaded {len(sft_data)} samples")

    print("\nConverting to DPO format (English)...")
    print("Rejected types: 1) Only conclusion 2) Wrong answers 3) Incomplete explanations")
    dpo_data = convert_alpaca_to_dpo(sft_data)

    print(f"\nSaving DPO dataset to {output_path}...")
    save_jsonl(dpo_data, output_path)

    print(f"\nDone! Generated {len(dpo_data)} DPO samples")

    print("\n" + "=" * 60)
    print("Sample examples:")
    print("=" * 60)

    for i in [0, 1, 5, 10, 50, 100, 150, 200, 300, 400, 500]:
        if i < len(dpo_data):
            item = dpo_data[i]
            print(f"\n[Sample {i + 1}]")
            print(f"Instruction: {item['instruction'][:50]}...")
            print(f"Chosen:   {item['chosen'][:60]}...")
            print(f"Rejected: {item['rejected']}")


if __name__ == "__main__":
    main()
