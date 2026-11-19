# App smoke test

`python app.py` on the full 20k-complaint index (flan-t5-base, 8-core CPU), queried with gradio_client:

```python
import time
from gradio_client import Client
c = Client("http://127.0.0.1:7860/", verbose=False)
for q, p in [("Why are customers unhappy with buy now pay later services?", "All products"),
             ("What problems do people report with credit card late fees?", "Credit card"),
             ("Who won the football world cup?", "All products")]:
    t = time.time()
    ans, src = c.predict(q, p, 5, api_name="/ask")
    print("Q:", q, f"({time.time()-t:.1f}s)\nA:", ans, "\nSOURCES:", src[:600], "\n")
```

Output (sources truncated to 600 characters by the script):

```text
Q: Why are customers unhappy with buy now pay later services? (3.7s)
A: They make over $2000.00 in service charges alone. They take the money before they process the order. They don't pay off the full purchase over 12 months. They are allowing my merchants to get paid without delivering products.

**Sources:** 8186981, 10377333, 9128942, 6409880, 6276615

_retrieval 0.04s · generation 3.4s_ 
SOURCES: **[1] Complaint 8186981** · Buy Now, Pay Later (BNPL) · Getting a line of credit · Sezzle Inc. · 2024-01-20 · similarity 0.62 · relevance 0.76

*Finding:* They make over $2000.00 in service charges alone
> . If you cant answer my question, please let me know and state youll find the right party who can. Ive also attached my payment history - Ive never been late and have placed and obscene amount of orders with Sezzle. Due ...

**[2] Complaint 10377333** · Buy Now, Pay Later (BNPL) · Problem when making payments · Klarna AB · 2024-10-07 · similarity 0.62 · relevance 0.46

*Finding:* They take t 

Q: What problems do people report with credit card late fees? (1.5s)
A: The late fees hit their credit score and credit report.

**Sources:** 1574480, 4339924, 5742736

_retrieval 0.02s · generation 1.4s_ 
SOURCES: **[1] Complaint 5187540** · Credit card · Fees or interest · CAPITAL ONE FINANCIAL CORPORATION · 2022-02-05 · similarity 0.75 · relevance 0.17

*Not used: judged off-topic for this question.*
> . There should be a law on the number of late fees allowed on the credit card, and a law on notices sent to dispute these late fees instead of the outsourced building that takes calls giving wrong information. My cards w...

**[2] Complaint 1574480** · Credit card · Billing disputes · CITIBANK, N.A. · 2015-09-22 · similarity 0.73 · relevance 0.60

*Finding:* late fees
> I have been disputing late fees w 

Q: Who won the football world cup? (0.1s)
A: Not enough evidence in the retrieved complaints to answer this.

_retrieval 0.02s · generation 0.0s_ 
SOURCES: **[1] Complaint 18491091** · Personal loan · Charged fees or interest you didn't expect · OM Financial, LLC · 2026-01-06 · similarity 0.23
> I paid my loan in full on . Still took out automatic payments. Requested a refund have not received

**[2] Complaint 9517194** · Personal loan · Charged fees or interest you didn't expect · Transform Credit Inc. · 2024-07-15 · similarity 0.22
> . No loan and they get $5.00 Every month. Win for them, loss for me. Very unfair practice

**[3] Complaint 11952712** · Personal loan · Problem with the payoff process at the end of the loan · ACIMA CREDIT, LLC · 2 

```
