# laq

To implement our method, we will re-write llama to fit our method

* Do not use `past_key_value` any more, use self data structure instead.
* Do not support FlashAttn