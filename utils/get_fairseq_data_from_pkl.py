import pickle
import sys
import os


def process(input_filename, question_output_filename, program_output_filename):
    os.makedirs(os.path.dirname(program_output_filename), exist_ok=True)
    os.makedirs(os.path.dirname(question_output_filename), exist_ok=True)
    json_data = pickle.load(open(input_filename, 'rb'))
    with open(question_output_filename, 'w') as ques_f_out, open(program_output_filename, 'w') as prog_f_out:
        for example in json_data:
            question = example['question'][:-1].replace(',', ' ,')
            ques_f_out.write(question + '\n')       # remove the question mark
            program_list = example['nmn_program'][::-1]
            prog_f_out.write(' '.join(program_list) + '\n')


if __name__ == '__main__':
    process(*sys.argv[1:])      # params: input_filename, question_output_filename, program_output_filename
