import click
import json
from agent.graph_hybrid import app


@click.command()
@click.option("--batch", help="Input JSONL file")
@click.option("--out", help="Output JSONL file")
def main(batch, out):
    with open(batch, "r") as f:
        questions = [json.loads(line) for line in f]

    with open(out, "w", encoding="utf-8") as f_out:
        for q_item in questions:
            print(f"Processing: {q_item['id']}...")

            inputs = {
                "question": q_item["question"],
                "format_hint": q_item["format_hint"],
                "retries": 0,
            }

            try:
                output_state = app.invoke(inputs)

                response = {
                    "id": q_item["id"],
                    "final_answer": output_state.get("final_answer"),
                    "sql": output_state.get("sql", ""),
                    "confidence": (0.8 if not output_state.get("sql_error") else 0.0),
                    "explanation": output_state.get("explanation", ""),
                    "citations": output_state.get("citations", []),
                }

                f_out.write(json.dumps(response) + "\n")
                f_out.flush()
                print(f" > Saved result for {q_item['id']}")

            except Exception as e:
                print(f" ! ERROR on {q_item['id']}: {e}")

                error_response = {
                    "id": q_item["id"],
                    "error": str(e),
                    "status": "failed",
                }
                f_out.write(json.dumps(error_response) + "\n")
                f_out.flush()


if __name__ == "__main__":
    main()
