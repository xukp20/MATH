# MATH
## Generate captions for asy command
2023/7/8: use gpt-3.5-turbo to generate captions for the test set.

The instruction may could be better.

| MATH  | problems with asy | total | percentage |
| ----- | ----------------- | ----- | ---------- |
| train | 707               | 7500  | 0.094      |
| test  | 419               | 5000  | 0.0838     |

| Category | algebra | counting_and_probability | geometry | intermediate_algebra | number_theory | prealgebra | precalculus |
| -------- | ------- | ------------------------ | -------- | -------------------- | ------------- | ---------- | ----------- |
| train    | 55      | 58                       | 324      | 40                   | 2             | 177        | 51          |
| test     | 23      | 47                       | 188      | 29                   | 1             | 100        | 31          |



2023/7/11 

Results with captions of gpt models on 42 samples of prm800k

| model                    | raw LLaMa 65B | w gpt3.5 | w gpt4 | w text | w gpt 3.5 strip | w gpt4 strip | with text strip |
| ------------------------ | ------------- | -------- | ------ | ------ | --------------- | ------------ | --------------- |
| correct (same + recheck) | 1 + 1         | 2 + 4    | 0 + 4  | 1 + 4  | 1 + 1           | 1 + 1        | 1 + 2           |

| correct + recheck | raw                             | 3.5                                                          | 4                                                            | text                                                         | 3.5 strip                   | 4 strip                   | text strip                                              |
| ----------------- | ------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | --------------------------- | ------------------------- | ------------------------------------------------------- |
|                   | prealgebra_930  prealgebra_1512 | geometry_248 prealgebra_378 prealgebra_930 geometry_226 geometry_795 counting_and_probability_731 | counting_and_probability_282 geometry_226 counting_and_probability_731 prealgebra_914 | geometry_248  algebra_1349 geometry_226 counting_and_probability_731 prealgebra_1114 | prealgebra_930 geometry_226 | geometry_248 geometry_226 | geometry_283  geometry_183 counting_and_probability_731 |



419 MATH test with asy

| MATH              | raw LLaMa | w 3.5   |
| ----------------- | --------- | ------- |
| correct + recheck | 15+9      | 11 + 11 |
| fail              | 27        | 77      |

- LLaMa: 
  - algebra_489
    counting_and_probability_250
    counting_and_probability_281
    counting_and_probability_288
    counting_and_probability_328
    geometry_145
    geometry_15
    geometry_242
    geometry_267
    geometry_375
    geometry_47
    prealgebra_1040
    prealgebra_1375
    prealgebra_1507
    prealgebra_930
- GPT3.5-turbo
  - 



