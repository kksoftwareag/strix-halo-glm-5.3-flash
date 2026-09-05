# Geschwindigkeitsmessungen

### Footprint-Proben (eine Anfrage, 300 Token)

| Lauf | Preset | Ladezeit | Prefill | Decode | Draft-Akzeptanz | Bedarf (Δ MemAvailable) | Peak GTT | geschätzt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| probe-tkmtp-32k | tkmtp-agent | 79 s | 24,9 t/s | 17,3 t/s | 87% | 99,9 GiB | 97,2 GiB | 99,2 GiB |

### Mehrnutzer (gleichzeitige Streams, je 8k Prompt und 512 Token Ausgabe)

**tk-UD-IQ2_XXS-nomtp-20260905-214917**

| Streams | Σ Decode t/s | je Stream t/s | Prefill je Stream t/s | TTFT | inkl. Prefill t/s | Draft-Akzeptanz | Mix-ups | Fehler |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 10,5 | 10,5 | 149,7 | 72,2 s | 4,2 | – | 0 | 0 |
| 2 | 11,5 | 5,8 | 65,8 | 162,4 s | 4,1 | – | 0 | 0 |
| 4 | 9,1 | 2,3 | 40,2 | 311,3 s | 3,5 | – | 0 | 0 |

**tk-UD-IQ2_XXS-nomtp-20260905-225036**

| Streams | Σ Decode t/s | je Stream t/s | Prefill je Stream t/s | TTFT | inkl. Prefill t/s | Draft-Akzeptanz | Mix-ups | Fehler |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 10,1 | 10,1 | 150,7 | 71,7 s | 4,2 | – | 0 | 0 |

**tk-merged-UD-IQ2_XXS-mtp-20260905-224355**: Absturz: GGML_ASSERT(width == mtp_dsa_sel_width)

**tk-mtp-UD-IQ1_M-mtp-20260905-220638**: Absturz: GGML_ASSERT(width == mtp_dsa_sel_width)

**tk-mtp-UD-IQ1_M-mtp-20260905-221145**: Absturz: GGML_ASSERT(width == mtp_dsa_sel_width)

**tk-mtp-UD-IQ1_M-mtp-20260905-224157**: Absturz: GGML_ASSERT(width == mtp_dsa_sel_width)

**tk-mtp-UD-IQ1_M-mtp-20260905-224623**

| Streams | Σ Decode t/s | je Stream t/s | Prefill je Stream t/s | TTFT | inkl. Prefill t/s | Draft-Akzeptanz | Mix-ups | Fehler |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 16,5 | 16,5 | 137,7 | 6,3 s | 11,8 | 74% | 0 | 0 |

**tk-mtp-UD-IQ1_M-mtp-20260905-224812**

| Streams | Σ Decode t/s | je Stream t/s | Prefill je Stream t/s | TTFT | inkl. Prefill t/s | Draft-Akzeptanz | Mix-ups | Fehler |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 10,7 | 10,7 | 152,6 | 13,7 s | 6,8 | 78% | 0 | 0 |

**tk-mtp-UD-IQ1_M-mtp-20260905-225414**: Absturz: GGML_ASSERT(width == mtp_dsa_sel_width)

**tk-mtp-UD-IQ1_M-mtp-20260905-225607**: Absturz: GGML_ASSERT(width == mtp_dsa_sel_width)

**tk-mtp-UD-IQ1_M-mtp-20260905-225805**: Absturz: GGML_ASSERT(width == mtp_dsa_sel_width)

**tk-mtp-UD-IQ2_XXS-mtp-20260905-220938**: Absturz: GGML_ASSERT(width == mtp_dsa_sel_width)

**tk-mtp-UD-IQ2_XXS-mtp-20260905-223310**: Absturz: GGML_ASSERT(width == mtp_dsa_sel_width)

**unsloth-UD-IQ1_M-mtp-20260905-221346**

| Streams | Σ Decode t/s | je Stream t/s | Prefill je Stream t/s | TTFT | inkl. Prefill t/s | Draft-Akzeptanz | Mix-ups | Fehler |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 15,7 | 15,7 | 143,5 | 75,3 s | 4,8 | 82% | 0 | 0 |
| 2 | 15,2 | 7,6 | 62,4 | 171,3 s | 4,3 | 83% | 0 | 0 |
| 4 | 9,8 | 2,5 | 38,1 | 329,5 s | 3,5 | 84% | 0 | 0 |

**unsloth-UD-IQ1_M-nomtp-20260905-223843**

| Streams | Σ Decode t/s | je Stream t/s | Prefill je Stream t/s | TTFT | inkl. Prefill t/s | Draft-Akzeptanz | Mix-ups | Fehler |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 13,4 | 13,4 | 149,2 | 72,4 s | 4,6 | – | 0 | 0 |

**unsloth-UD-IQ2_XXS-mtp-20260905-223518**

| Streams | Σ Decode t/s | je Stream t/s | Prefill je Stream t/s | TTFT | inkl. Prefill t/s | Draft-Akzeptanz | Mix-ups | Fehler |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 14,3 | 14,3 | 146,6 | 73,7 s | 4,7 | 79% | 0 | 0 |

**unsloth-UD-IQ2_XXS-mtp-n3-20260905-231320**

| Streams | Σ Decode t/s | je Stream t/s | Prefill je Stream t/s | TTFT | inkl. Prefill t/s | Draft-Akzeptanz | Mix-ups | Fehler |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 14,8 | 14,8 | 146,6 | 73,7 s | 4,7 | 78% | 0 | 0 |

**unsloth-UD-IQ2_XXS-mtp-n4-20260905-231638**

| Streams | Σ Decode t/s | je Stream t/s | Prefill je Stream t/s | TTFT | inkl. Prefill t/s | Draft-Akzeptanz | Mix-ups | Fehler |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 15,2 | 15,2 | 146,6 | 73,7 s | 4,8 | 82% | 0 | 0 |

**unsloth-UD-IQ2_XXS-nomtp-20260905-230101**

| Streams | Σ Decode t/s | je Stream t/s | Prefill je Stream t/s | TTFT | inkl. Prefill t/s | Draft-Akzeptanz | Mix-ups | Fehler |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 13,0 | 13,0 | 151,7 | 71,2 s | 4,6 | – | 0 | 0 |

