import src.rag as rag

while True:
    question = input(f"Câu hỏi:")
    output = rag.rag(question)
    print("---" * 30)
