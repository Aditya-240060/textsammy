import time
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class TextSummarizer:
    def __init__(self, model_name: str = "sshleifer/distilbart-cnn-12-6"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"[Summarizer] Loading '{self.model_name}' on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)
        
        # Put the neural network in evaluation mode (disables dropout layers)
        self.model.eval()
        print("[Summarizer] Model loaded and ready.")

    def _split_into_chunks(self, text: str, max_words_per_chunk: int = 650) -> list[str]:
        """
        Splits a long article into smaller chunks along sentence boundaries ('. ')
        so we never exceed the Transformer's 1024-token limit or cut a sentence in half.
        """
        sentences = text.replace("\n", " ").split(". ")
        chunks = []
        current_chunk = []
        current_word_count = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            # Add the period back if it was removed by split
            if not sentence.endswith("."):
                sentence += "."

            word_count = len(sentence.split())

            # If adding this sentence pushes us over the limit, save current chunk and start a new one
            if current_word_count + word_count > max_words_per_chunk and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_word_count = word_count
            else:
                current_chunk.append(sentence)
                current_word_count += word_count

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def _summarize_single_chunk(self, chunk: str, min_length: int, max_length: int) -> str:
        """Runs PyTorch inference on a single chunk of text."""
        inputs = self.tokenizer(
            chunk,
            max_length=1024,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        input_token_len = inputs["input_ids"].shape[1]

        # Guardrail: Prevent min_length/max_length from exceeding the actual input length
        safe_max_len = min(max_length, max(20, int(input_token_len * 0.8)))
        safe_min_len = min(min_length, max(10, int(safe_max_len * 0.5)))

        with torch.no_grad():
            summary_ids = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                num_beams=4,
                length_penalty=2.0,
                min_length=safe_min_len,
                max_length=safe_max_len,
                early_stopping=True
            )

        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    def summarize(self, text: str, min_length: int = 30, max_length: int = 130) -> dict:
        """
        Main public method that chunks long text, summarizes each chunk,
        and returns both the summary and analytics metadata for our SQL database.
        """
        start_time = time.perf_counter()
        cleaned_text = text.strip()

        if not cleaned_text:
            raise ValueError("Input text cannot be empty.")

        chunks = self._split_into_chunks(cleaned_text)
        
        # Adjust per-chunk length if the article was split into multiple chunks
        chunk_min_len = max(15, min_length // len(chunks))
        chunk_max_len = max(40, max_length // len(chunks))

        chunk_summaries = [
            self._summarize_single_chunk(chunk, chunk_min_len, chunk_max_len)
            for chunk in chunks
        ]

        final_summary = " ".join(chunk_summaries)
        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return {
            "summary_text": final_summary,
            "model_used": self.model_name,
            "num_chunks": len(chunks),
            "original_word_count": len(cleaned_text.split()),
            "summary_word_count": len(final_summary.split()),
            "processing_time_ms": elapsed_ms
        }