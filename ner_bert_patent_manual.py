"""This recipe requires Prodigy v1.10+."""
from typing import List, Optional, Union, Iterable, Dict, Any
from tokenizers import BertWordPieceTokenizer
from prodigy.components.loaders import get_stream
from prodigy.util import get_labels
import prodigy
from transformers import (AutoModelForTokenClassification, 
                          AutoTokenizer, 
                          pipeline,
                          )


@prodigy.recipe(
    "bert.ner.manual",
    # fmt: off
    dataset=("Dataset to save annotations to", "positional", None, str),
    source=("Data to annotate (file path or '-' to read from standard input)", "positional", None, str),
    loader=("Loader (guessed from file extension if not set)", "option", "lo", str),
    label=("Comma-separated label(s) to annotate or text file with one label per line", "option", "l", get_labels),
    tokenizer_vocab=("Tokenizer vocab file", "option", "tv", str),
    lowercase=("Set lowercase=True for tokenizer", "flag", "LC", bool),
    hide_special=("Hide SEP and CLS tokens visually", "flag", "HS", bool),
    hide_wp_prefix=("Hide wordpieces prefix like ##", "flag", "HW", bool)
    # fmt: on
)
def ner_manual_tokenizers_bert(
    dataset: str,
    source: Union[str, Iterable[dict]],
    loader: Optional[str] = None,
    label: Optional[List[str]] = None,
    tokenizer_vocab: Optional[str] = None,
    lowercase: bool = False,
    hide_special: bool = False,
    hide_wp_prefix: bool = False,
) -> Dict[str, Any]:
    """Example recipe that shows how to use model-specific tokenizers like the
    BERT word piece tokenizer to preprocess your incoming text for fast and
    efficient NER annotation and to make sure that all annotations you collect
    always map to tokens and can be used to train and fine-tune your model
    (even if the tokenization isn't that intuitive, because word pieces). The
    selection automatically snaps to the token boundaries and you can double-click
    single tokens to select them.

    Setting "honor_token_whitespace": true will ensure that whitespace between
    tokens is only shown if whitespace is present in the original text. This
    keeps the text readable.

    Requires Prodigy v1.10+ and usese the HuggingFace tokenizers library."""
    stream = get_stream(source, loader=loader, input_key="text")
    
    model_checkpoint = "ktgiahieu/bert-for-patents-finetuned-ner"
    model = AutoModelForTokenClassification.from_pretrained(model_checkpoint)                                                        
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    model_pipeline = pipeline(task="ner", model=model, tokenizer=tokenizer)
    # Define the ClassLabel feature with the label names
    label_list = ['B-Activity', 'B-Administration', 'B-Age', 'B-Area', 'B-Biological_attribute', 'B-Biological_structure', 'B-Clinical_event', 'B-Color', 'B-Coreference', 'B-Date', 'B-Detailed_description', 'B-Diagnostic_procedure', 'B-Disease_disorder', 'B-Distance', 'B-Dosage', 'B-Duration', 'B-Family_history', 'B-Frequency', 'B-Height', 'B-History', 'B-Lab_value', 'B-Mass', 'B-Medication', 'B-Nonbiological_location', 'B-Occupation', 'B-Other_entity', 'B-Other_event', 'B-Outcome', 'B-Personal_background', 'B-Qualitative_concept', 'B-Quantitative_concept', 'B-Severity', 'B-Sex', 'B-Shape', 'B-Sign_symptom', 'B-Subject', 'B-Texture', 'B-Therapeutic_procedure', 'B-Time', 'B-Volume', 'B-Weight', 'I-Activity', 'I-Administration', 'I-Age', 'I-Area', 'I-Biological_attribute', 'I-Biological_structure', 'I-Clinical_event', 'I-Color', 'I-Coreference', 'I-Date', 'I-Detailed_description', 'I-Diagnostic_procedure', 'I-Disease_disorder', 'I-Distance', 'I-Dosage', 'I-Duration', 'I-Family_history', 'I-Frequency', 'I-Height', 'I-History', 'I-Lab_value', 'I-Mass', 'I-Medication', 'I-Nonbiological_location', 'I-Occupation', 'I-Other_entity', 'I-Other_event', 'I-Outcome', 'I-Personal_background', 'I-Qualitative_concept', 'I-Quantitative_concept', 'I-Severity', 'I-Shape', 'I-Sign_symptom', 'I-Subject', 'I-Texture', 'I-Therapeutic_procedure', 'I-Time', 'I-Volume', 'I-Weight', 'O']
    # label_list_truncated = [x[2:] if x!='O' else 'O' for x in label_list]
    keep_list = ['Diagnostic_procedure', 'Medication', 'Lab_value', 'Detailed_description']
    # You can replace this with other tokenizers if needed
    # tokenizer = BertWordPieceTokenizer(tokenizer_vocab, lowercase=lowercase)
    sep_token = tokenizer.sep_token
    cls_token = tokenizer.cls_token
    special_tokens = (sep_token, cls_token)
    wp_prefix = '##'

    def add_tokens(stream):
        for eg in stream:
            tokens = tokenizer(eg["text"], return_offsets_mapping=True, return_special_tokens_mask=True)
            eg_tokens = []
            idx = 0
            for (text, (start, end), tid) in zip(
                [tokenizer.decode(x) for x in tokens["input_ids"]], tokens["offset_mapping"], tokens["input_ids"]
            ):
                # If we don't want to see special tokens, don't add them
                if hide_special and text in special_tokens:
                    continue
                # If we want to strip out word piece prefix, remove it from text
                if hide_wp_prefix and wp_prefix is not None:
                    if text.startswith(wp_prefix):
                        text = text[len(wp_prefix) :]
                token = {
                    "text": text,
                    "id": idx,
                    "start": start,
                    "end": end,
                    # This is the encoded ID returned by the tokenizer
                    "tokenizer_id": tid,
                    # Don't allow selecting spacial SEP/CLS tokens
                    "disabled": text in special_tokens,
                }
                eg_tokens.append(token)
                idx += 1
            for i, token in enumerate(eg_tokens):
                # If the next start offset != the current end offset, we
                # assume there's whitespace in between
                if i < len(eg_tokens) - 1 and token["text"] not in special_tokens:
                    next_token = eg_tokens[i + 1]
                    token["ws"] = (
                        next_token["start"] > token["end"]
                        or next_token["text"] in special_tokens
                    )
                else:
                    token["ws"] = True
            eg["tokens"] = eg_tokens


            # Get the entities from the model
            orig_entities = model_pipeline(eg["text"])
        
            entities = []
            for i in range(len(orig_entities)):
                orig_entities[i]['entity'] = label_list[int(orig_entities[i]['entity'][6:])]
                if orig_entities[i]['entity'] == 'O' or orig_entities[i]['entity'][2:] not in keep_list:
                    continue
                entities.append(orig_entities[i])

            entities_markup = []
            for i in range(len(entities)):
                if len(entities_markup) == 0:
                    entities_markup.append([entities[i]['start'], entities[i]['end'], entities[i]['entity'][2:]])
                    continue
                if (entities[i]['start'] == entities[i-1]['end'] \
                    or entities[i]['start'] == entities[i-1]['end']+1 \
                    # or entities[i]['start'] == entities[i-1]['end']+2 \
                    # or entities[i]['start'] == entities[i-1]['end']+3 \
                ) and \
                    entities[i]['entity'][2:] == entities[i-1]['entity'][2:]:
                    entities_markup[-1][1] = entities[i]['end']
                else:
                    entities_markup.append([entities[i]['start'], entities[i]['end'], entities[i]['entity'][2:]])

            spans = []
            for ent in entities_markup:
                # Create span dict for the predicted entity
                # We need to map the start/end offsets to the token IDs
                # so we can use the span in the UI

                # Find the first token that starts after the entity start
                start_token = next(
                    (t for t in eg_tokens if t["start"] >= ent[0]), None
                )
                # Find the last token that ends before the entity end
                for i in range(len(eg_tokens)-2, -1, -1):
                    if eg_tokens[i]['end'] <= ent[1]:
                        end_token = eg_tokens[i]
                        break
                # end_token = next(
                #     (t for t in reversed(eg_tokens) if t["end"] <= ent[1]), None
                # )
                # If we can't find a token for the start/end, skip this entity
                if start_token is None or end_token is None:
                    continue
                # If the start token is after the end token, skip this entity
                if start_token["id"] > end_token["id"]:
                    continue

                spans.append({
                    "token_start": start_token["id"],
                    "token_end": end_token["id"],
                    "label": ent[2],
                    "start": ent[0],
                    "end": ent[1],
                })
                
            eg["spans"] = spans
            yield eg

    stream = add_tokens(stream)

    return {
        "dataset": dataset,
        "stream": stream,
        "view_id": "ner_manual",
        "config": {
            "honor_token_whitespace": True,
            "labels": keep_list,
            "exclude_by": "input",
            "force_stream_order": True,
        },
    }