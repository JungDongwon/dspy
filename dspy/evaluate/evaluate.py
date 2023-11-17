from openai import InvalidRequestError
from openai.error import APIError

import dsp
import tqdm
import threading
import pandas as pd
from collections import Counter

from IPython.display import display as ipython_display, HTML
from concurrent.futures import ThreadPoolExecutor, as_completed

from dsp.utils import EM, F1, HotPotF1
from dsp.evaluation.utils import *

"""
TODO: Counting failures and having a max_failure count. When that is exceeded (also just at the end),
we print the number of failures, the first N examples that failed, and the first N exceptions raised.
"""


class Evaluate:
    def __init__(self, *, devset, outfile, metric=None, num_threads=1, display_progress=False,
                 display=True, max_errors=5):
        self.devset = devset
        self.outfile = outfile
        self.metric = metric
        self.num_threads = num_threads
        self.display_progress = display_progress
        self.display = display
        self.max_errors = max_errors
        self.error_count = 0
        self.error_lock = threading.Lock()

    def _execute_single_thread(self, wrapped_program, devset, display_progress):
        total_score = Counter()
        ntotal = 0
        reordered_devset = []
        
        pbar = tqdm.tqdm(total=len(devset), dynamic_ncols=True, disable=not display_progress)
        for idx, arg in devset:
            example_idx, example, prediction, score = wrapped_program(idx, arg)
            reordered_devset.append((example_idx, example, prediction, score))
            total_score += score
            if not total_score:
                total_score = Counter(score)
            ntotal += 1
            self._update_progress(pbar, total_score, ntotal)
        pbar.close()
        
        return reordered_devset, total_score, ntotal

    def _execute_multi_thread(self, wrapped_program, devset, num_threads, display_progress):
        total_score = Counter()
        ntotal = 0
        reordered_devset = []
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = {executor.submit(wrapped_program, idx, arg) for idx, arg in devset}
            pbar = tqdm.tqdm(total=len(devset), dynamic_ncols=True, disable=not display_progress)

            for future in as_completed(futures):
                example_idx, example, prediction, score = future.result()
                reordered_devset.append((example_idx, example, prediction, score))
                total_score += score
                ntotal += 1
                self._update_progress(pbar, score, ntotal)
            pbar.close()

        return reordered_devset, total_score, ntotal

    def _update_progress(self, pbar, score, ntotal):
        try:
            metric_name = list(score.keys())[0] # display the first score
        except:
            print(score)
        pbar.set_description(f"Average {metric_name}: {score[metric_name]} / {ntotal}  ({round(100 * score[metric_name] / ntotal, 1)}%)")
        pbar.update()

    def __call__(self, program, metric=None, devset=None, num_threads=None,
                 display_progress=None, display=None,
                 return_all_scores=False):
        metric = metric if metric is not None else self.metric
        devset = devset if devset is not None else self.devset
        num_threads = num_threads if num_threads is not None else self.num_threads
        display_progress = display_progress if display_progress is not None else self.display_progress

        display = self.display if display is None else display
        display_progress = display_progress and display

        def wrapped_program(example_idx, example):
            # NOTE: TODO: Won't work if threads create threads!
            creating_new_thread = threading.get_ident() not in dsp.settings.stack_by_thread
            if creating_new_thread:
                dsp.settings.stack_by_thread[threading.get_ident()] = list(dsp.settings.main_stack)
                # print(threading.get_ident(), dsp.settings.stack_by_thread[threading.get_ident()])

            # print(type(example), example)

            try:
                prediction = program(**example.inputs())
                score = metric(example, prediction)  # FIXME: TODO: What's the right order? Maybe force name-based kwargs!
                return example_idx, example, prediction, score
            except Exception as e:
                with self.error_lock:
                    self.error_count += 1
                    current_error_count = self.error_count
                if current_error_count >= self.max_errors:
                    raise e
                print(f"Error for example in dev set: \t\t {e}")
                return example_idx, example, dict(), 0.0
            finally:
                if creating_new_thread:
                    del dsp.settings.stack_by_thread[threading.get_ident()]

        devset = list(enumerate(devset))

        if num_threads == 1:
            reordered_devset, total_score, ntotal = self._execute_single_thread(wrapped_program, devset, display_progress)
        else:
            reordered_devset, total_score, ntotal = self._execute_multi_thread(wrapped_program, devset, num_threads, display_progress)

        if display:
            for metric_name, score in total_score.items():
                print(f"Average {metric_name}: {score} / {ntotal}  ({round(100 * score / ntotal, 1)}%)")

        predicted_devset = sorted(reordered_devset)

        # data = [{**example, **prediction, 'correct': score} for example, prediction, score in zip(reordered_devset, preds, scores)]
        #data = [merge_dicts(example, prediction) | {'correct': score} for _, example, prediction, score in predicted_devset]
        data = [merge_dicts(merge_dicts(example, prediction), score) for _, example, prediction, score in predicted_devset]
        
        df = pd.DataFrame(data)

        # Truncate every cell in the DataFrame
        df = df.map(truncate_cell)

        # Rename the 'correct' column to the name of the metric function
        #metric_name = metric.__name__
        #df.rename(columns={'correct': metric_name}, inplace=True)
        df.to_csv(self.outfile)
                
        if return_all_scores:
            return {metric_name: round(100 * score / ntotal, 2) for metric_name, score in total_score.items()}, [score for *_, score in predicted_devset]

        return {metric_name: round(100 * score / ntotal, 2) for metric_name, score in total_score.items()}

def merge_dicts(d1, d2):
    merged = {}
    for k, v in d1.items():
        if k in d2:
            merged[f"example_{k}"] = v
        else:
            merged[k] = v

    for k, v in d2.items():
        if k in d1:
            merged[f"pred_{k}"] = v
        else:
            merged[k] = v

    return merged


def truncate_cell(content):
    """Truncate content of a cell to 25 words."""
    words = str(content).split()
    if len(words) > 25:
        return ' '.join(words[:25]) + '...'
    return content

def configure_dataframe_display(df, metric_name):
    """Set various pandas display options for DataFrame."""
    pd.options.display.max_colwidth = None
    pd.set_option('display.max_colwidth', 15)  # Adjust the number as needed
    pd.set_option('display.width', 400)  # Adjust

    # df[metric_name] = df[metric_name].apply(lambda x: f'✔️ [{x}]' if x is True else f'❌ [{x}]')
    df.loc[:, metric_name] = df[metric_name].apply(lambda x: f'✔️ [{x}]' if x is True else f'❌ [{x}]')

    # Return styled DataFrame
    return df.style.set_table_styles([
        {'selector': 'th', 'props': [('text-align', 'left')]},
        {'selector': 'td', 'props': [('text-align', 'left')]}
    ]).set_properties(**{
        'text-align': 'left',
        'white-space': 'pre-wrap',
        'word-wrap': 'break-word',
        'max-width': '400px'
    })

# FIXME: TODO: The merge_dicts stuff above is way too quick and dirty.