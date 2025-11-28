import dspy


class RouterSignature(dspy.Signature):
    """Classify the user query to decide the tool.
    - 'sql': precise data questions, aggregations, counting, or looking up DB records.
    - 'rag': questions about policies, marketing definitions, return windows, or text docs.
    - 'hybrid': requires both data lookup AND policy definitions (e.g. "Revenue during Summer Sale 1997").
    """

    question = dspy.InputField()
    classification = dspy.OutputField(
        desc="EXACTLY one of: sql, rag, hybrid. No other text."
    )


class Router(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought(RouterSignature)

    def forward(self, question):
        return self.prog(question=question)


class Planner(dspy.Signature):
    """
    Analyzes the retrieved documents and user question to extract critical constraints
    and parameters needed for SQL generation or final RAG synthesis.

    Extract specific dates, date ranges, KPI definitions, and product categories.
    """

    question = dspy.InputField(desc="The user's original question.")
    retrieved_context = dspy.InputField(desc="Relevant documents chunks from RAG.")

    constraints = dspy.OutputField(
        desc="A concise summary of all extracted constraints: date ranges (e.g., '1997-06-01 to 1997-06-30'), KPI formulas, and relevant entities/categories."
    )


class TextToSQL(dspy.Signature):
    """Generate executable SQLite query for the Northwind database.
    USE VALID TABLES AND COLUMNS NAMES.
    Rules:
    1. Revenue Formula: SUM(UnitPrice * Quantity * (1 - Discount)) from 'order_items'
    2. Dates: SQLite uses strings ('YYYY-MM-DD'). Use strftime('%Y', OrderDate) for years.
    3. Gross Margin: If Cost is missing, use SUM(0.3 * UnitPrice * Quantity * (1 - Discount)) since CostOfGoods ≈ 0.7 * UnitPrice, so margin factor is 0.3.
    4. If needed, map categories via Categories join through products.CategoryID.
    5. Prefer orders + "order_items" + products joins.
    """

    question = dspy.InputField()
    db_schema = dspy.InputField(desc="Table schema with columns")
    constraints = dspy.InputField(desc="Context, date ranges, or KPI formulas from RAG")
    error_feedback = dspy.InputField(
        desc="Error from previous run to fix", optional=True
    )
    sql_query = dspy.OutputField(desc="A single valid SQLite query string")


class Synthesizer(dspy.Signature):
    """Answer the question based on the tools output.
    Ensure the final_answer matches the format_hint EXACTLY (e.g., if format_hint is 'int', return only the number).
    """

    question = dspy.InputField()
    sql_query = dspy.InputField()
    sql_result = dspy.InputField(desc="Rows returned from database")
    retrieved_context = dspy.InputField(desc="Relevant text chunks from docs")
    format_hint = dspy.InputField(desc="The required output type/format")

    final_answer = dspy.OutputField(desc="The answer matching format_hint")
    citations = dspy.OutputField(desc="List of DB tables and Doc Chunk IDs used")
    explanation = dspy.OutputField(desc="Brief justification < 2 sentences")
