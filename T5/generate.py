import time
import torch
from torch.utils.data import DataLoader, SequentialSampler


def generate_sql(model, eval_dataset, args, collator, verbose=0):

    file_name = args.config.split('/')[-1]
    start_time = time.time()
    eval_sampler = SequentialSampler(eval_dataset)
    dataloader = DataLoader(
                                eval_dataset, 
                                sampler=eval_sampler, 
                                batch_size=args.eval_batch_size, 
                                drop_last=False,
                                collate_fn=collator
                                )
    tokenizer = eval_dataset.tokenizer
    model.eval()
    

    do_sample = True if args.num_beams == 1 else False
    with torch.no_grad():

        out_eval = {}
        for idx, batch in enumerate(dataloader, 1):

            input_ids = batch['inputs'].to(args.device)
            labels = batch['labels'].to(args.device)
            db_ids = batch['db_id']
            is_impossibles = batch['is_impossible']
            data_ids = batch['id']
            
            generation_output = model.generate(
                                    input_ids=input_ids, 
                                    num_beams=args.num_beams,
                                    max_length=args.max_length,
                                    do_sample=do_sample,
                                    num_return_sequences=args.num_samples,
                                    repetition_penalty=args.repetition_penalty,
                                    length_penalty=args.length_penalty,
                                    early_stopping=args.early_stopping, 
                                    return_dict_in_generate=True, 
                                    output_scores=True
                                    )

            preds = generation_output['sequences'].cpu() if args.device == 'cuda' else generation_output['sequences']
            sequences_scores = generation_output['sequences_scores'].cpu() if args.device == 'cuda' else generation_output['sequences_scores']
            logits = torch.stack(generation_output['scores'], dim=1)[::int(args.num_beams/args.num_samples)]
            logits = logits.cpu() if args.device == 'cuda' else logits
            output_prob = torch.softmax(logits, dim=2)
            log_prob = torch.log_softmax(logits, dim=2)
            sequences_entropy = ( torch.sum(output_prob * log_prob, dim=2) * (-1) ).numpy()

            for i in range(len(preds)):
                
                text = tokenizer.decode(input_ids[i], skip_special_tokens=True)
                real = tokenizer.decode(labels[i], skip_special_tokens=True)
                pred = tokenizer.decode(preds[i], skip_special_tokens=True)

                pred_tensor = preds[i][1:]
                entropy = sequences_entropy[i].tolist()
                if tokenizer.eos_token_id in pred_tensor:
                    pred_eos_idx = torch.nonzero(pred_tensor==tokenizer.eos_token_id)[0].item()
                    entropy = entropy[:pred_eos_idx+1]
                result = {}
                result['question'] = text
                result['real'] = real
                result['pred'] = pred
                result['db_id'] = db_ids[i]
                result['is_impossible'] = is_impossibles[i]
                result['sequence_entropy'] = tuple(entropy)
                out_eval[data_ids[i]] = result

            if verbose>0:
                print(f'{idx}/{len(dataloader)} ({round(idx/len(dataloader)*100, 4)}%) --- {file_name}', end='\r')

    if verbose>0:
        print(f"inference took {round(time.time() - start_time, 6)} secs")

    return out_eval