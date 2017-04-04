
for epoch_num in range(10):
    
    step = 0
    
    for file_ind, (file_name, source) in enumerate(tqdm(train_gen())):
        
        # at the begining of the file, reset hidden to zero
        egdt.init_hidden_(random=False)
            
        for source_ in batch_gen(seq_length, source):
            
            
            step += 1
            
            input_source, punctuation_target = extract_punc(source_, egdt.char2vec.chars, egdt.output_char2vec.chars)
            #print(len(input_source), len(punctuation_target))
            
            try:
                egdt.forward(input_source, punctuation_target)
                if step%1 == 0:
                    egdt.descent()
                    
            except KeyError:
                print(source)
                raise KeyError
            

            if step%500 == 499:
                clear_output(wait=True)
                print('Epoch {:d}'.format(epoch_num))

                egdt.softmax_()

                plot_progress(egdt.embeded[:130].data.numpy(), 
                              egdt.output[:20].data.numpy(), 
                              egdt.softmax[:20].data.numpy(),
                              egdt.losses)
                
                punctuation_output = egdt.output_chars()
                result = apply_punc(input_source, punctuation_output)
                
                # print(punctuation_output, punctuation_target)
                print(result + "\n")
                
                print_pc(punctuation_output, punctuation_target)
                
        # validation, ran once in a while. takes a munite to run.
        if file_ind%200 == 1999:
            print('Dev Set Performance {:d}'.format(epoch_num))
