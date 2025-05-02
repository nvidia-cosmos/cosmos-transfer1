# Open Source License Attribution

   Cosmos uses Open Source components. You can find the details of these open-source projects along with license information below, sorted alphabetically.
   We are grateful to the developers for their contributions to open source and acknowledge these below.

## Better-Profanity - [MIT License](https://github.com/snguyenthanh/better_profanity/blob/master/LICENSE)

   ```

   Copyright (c) 2018 The Python Packaging Authority

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in all
   copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
   SOFTWARE.

   ```

## FFmpeg - [FFMPEG License](https://github.com/FFmpeg/FFmpeg/blob/master/LICENSE.md)

   ```
   # License

   Most files in FFmpeg are under the GNU Lesser General Public License version 2.1
   or later (LGPL v2.1+). Read the file `COPYING.LGPLv2.1` for details. Some other
   files have MIT/X11/BSD-style licenses. In combination the LGPL v2.1+ applies to
   FFmpeg.

   Some optional parts of FFmpeg are licensed under the GNU General Public License
   version 2 or later (GPL v2+). See the file `COPYING.GPLv2` for details. None of
   these parts are used by default, you have to explicitly pass `--enable-gpl` to
   configure to activate them. In this case, FFmpeg's license changes to GPL v2+.

   Specifically, the GPL parts of FFmpeg are:

   - libpostproc
   - optional x86 optimization in the files
       - `libavcodec/x86/flac_dsp_gpl.asm`
       - `libavcodec/x86/idct_mmx.c`
       - `libavfilter/x86/vf_removegrain.asm`
   - the following building and testing tools
       - `compat/solaris/make_sunver.pl`
       - `doc/t2h.pm`
       - `doc/texi2pod.pl`
       - `libswresample/tests/swresample.c`
       - `tests/checkasm/*`
       - `tests/tiny_ssim.c`
   - the following filters in libavfilter:
       - `signature_lookup.c`
       - `vf_blackframe.c`
       - `vf_boxblur.c`
       - `vf_colormatrix.c`
       - `vf_cover_rect.c`
       - `vf_cropdetect.c`
       - `vf_delogo.c`
       - `vf_eq.c`
       - `vf_find_rect.c`
       - `vf_fspp.c`
       - `vf_histeq.c`
       - `vf_hqdn3d.c`
       - `vf_kerndeint.c`
       - `vf_lensfun.c` (GPL version 3 or later)
       - `vf_mcdeint.c`
       - `vf_mpdecimate.c`
       - `vf_nnedi.c`
       - `vf_owdenoise.c`
       - `vf_perspective.c`
       - `vf_phase.c`
       - `vf_pp.c`
       - `vf_pp7.c`
       - `vf_pullup.c`
       - `vf_repeatfields.c`
       - `vf_sab.c`
       - `vf_signature.c`
       - `vf_smartblur.c`
       - `vf_spp.c`
       - `vf_stereo3d.c`
       - `vf_super2xsai.c`
       - `vf_tinterlace.c`
       - `vf_uspp.c`
       - `vf_vaguedenoiser.c`
       - `vsrc_mptestsrc.c`

   Should you, for whatever reason, prefer to use version 3 of the (L)GPL, then
   the configure parameter `--enable-version3` will activate this licensing option
   for you. Read the file `COPYING.LGPLv3` or, if you have enabled GPL parts,
   `COPYING.GPLv3` to learn the exact legal terms that apply in this case.

   There are a handful of files under other licensing terms, namely:

   * The files `libavcodec/jfdctfst.c`, `libavcodec/jfdctint_template.c` and
     `libavcodec/jrevdct.c` are taken from libjpeg, see the top of the files for
     licensing details. Specifically note that you must credit the IJG in the
     documentation accompanying your program if you only distribute executables.
     You must also indicate any changes including additions and deletions to
     those three files in the documentation.
   * `tests/reference.pnm` is under the expat license.


   ## External libraries

   FFmpeg can be combined with a number of external libraries, which sometimes
   affect the licensing of binaries resulting from the combination.

   ### Compatible libraries

   The following libraries are under GPL version 2:
   - avisynth
   - frei0r
   - libcdio
   - libdavs2
   - librubberband
   - libvidstab
   - libx264
   - libx265
   - libxavs
   - libxavs2
   - libxvid

   When combining them with FFmpeg, FFmpeg needs to be licensed as GPL as well by
   passing `--enable-gpl` to configure.

   The following libraries are under LGPL version 3:
   - gmp
   - libaribb24
   - liblensfun

   When combining them with FFmpeg, use the configure option `--enable-version3` to
   upgrade FFmpeg to the LGPL v3.

   The VMAF, mbedTLS, RK MPI, OpenCORE and VisualOn libraries are under the Apache License
   2.0. That license is incompatible with the LGPL v2.1 and the GPL v2, but not with
   version 3 of those licenses. So to combine these libraries with FFmpeg, the
   license version needs to be upgraded by passing `--enable-version3` to configure.

   The smbclient library is under the GPL v3, to combine it with FFmpeg,
   the options `--enable-gpl` and `--enable-version3` have to be passed to
   configure to upgrade FFmpeg to the GPL v3.

   ### Incompatible libraries

   There are certain libraries you can combine with FFmpeg whose licenses are not
   compatible with the GPL and/or the LGPL. If you wish to enable these
   libraries, even in circumstances that their license may be incompatible, pass
   `--enable-nonfree` to configure. This will cause the resulting binary to be
   unredistributable.

   The Fraunhofer FDK AAC and OpenSSL libraries are under licenses which are
   incompatible with the GPLv2 and v3. To the best of our knowledge, they are
   compatible with the LGPL.

   ```

## Hydra-core [MIT License](https://github.com/facebookresearch/hydra/blob/main/LICENSE)

   ```

   MIT License

   Copyright (c) Facebook, Inc. and its affiliates.

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in all
   copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
   SOFTWARE.

   ```

## ImageIo - [BSD 2-Clause "Simplified" License](https://github.com/imageio/imageio/blob/master/LICENSE)

   ```

   Copyright (c) 2014-2022, imageio developers
   All rights reserved.

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are met:

   * Redistributions of source code must retain the above copyright notice, this
     list of conditions and the following disclaimer.

   * Redistributions in binary form must reproduce the above copyright notice,
     this list of conditions and the following disclaimer in the documentation
     and/or other materials provided with the distribution.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
   AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
   IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
   DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
   FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
   DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
   SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
   CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
   OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

   ```

## Iopath - [MIT License](https://github.com/facebookresearch/iopath/blob/main/LICENSE)

   ```
   MIT License

   Copyright (c) Facebook, Inc. and its affiliates.

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in all
   copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
   SOFTWARE.

   ```

## Llama-Guard-3-8B [META LLAMA 3 COMMUNITY LICENSE](https://github.com/meta-llama/llama3/blob/main/LICENSE)

   ```

   META LLAMA 3 COMMUNITY LICENSE AGREEMENT

   Meta Llama 3 Version Release Date: April 18, 2024

   “Agreement” means the terms and conditions for use, reproduction, distribution, and
   modification of the Llama Materials set forth herein.

   “Documentation” means the specifications, manuals, and documentation accompanying Meta
   Llama 3 distributed by Meta at https://llama.meta.com/get-started/.

   “Licensee” or “you” means you, or your employer or any other person or entity (if you are
   entering into this Agreement on such person or entity’s behalf), of the age required under
   applicable laws, rules, or regulations to provide legal consent and that has legal authority
   to bind your employer or such other person or entity if you are entering into this Agreement
   on their behalf.

   “Meta Llama 3” means the foundational large language models and software and algorithms,
   including machine-learning model code, trained model weights, inference-enabling code,
   training-enabling code, fine-tuning-enabling code, and other elements of the foregoing
   distributed by Meta at https://llama.meta.com/llama-downloads.

   “Llama Materials” means, collectively, Meta’s proprietary Meta Llama 3 and Documentation
   (and any portion thereof) made available under this Agreement.

   “Meta” or “we” means Meta Platforms Ireland Limited (if you are located in or, if you are
   an entity, your principal place of business is in the EEA or Switzerland) and Meta Platforms,
   Inc. (if you are located outside of the EEA or Switzerland).

   By clicking “I Accept” below or by using or distributing any portion or element of the Llama
   Materials, you agree to be bound by this Agreement.

   1. License Rights and Redistribution.

   a. Grant of Rights. You are granted a non-exclusive, worldwide, non-transferable and
   royalty-free limited license under Meta’s intellectual property or other rights owned by
   Meta embodied in the Llama Materials to use, reproduce, distribute, copy, create derivative
   works of, and make modifications to the Llama Materials.

   b. Redistribution and Use.
      i. If you distribute or make available the Llama Materials (or any derivative works
      thereof), or a product or service that uses any of them, including another AI model, you
      shall (A) provide a copy of this Agreement with any such Llama Materials; and (B)
      prominently display “Built with Meta Llama 3” on a related website, user interface,
      blogpost, about page, or product documentation. If you use the Llama Materials to create,
      train, fine tune, or otherwise improve an AI model, which is distributed or made available,
      you shall also include “Llama 3” at the beginning of any such AI model name.

      ii. If you receive Llama Materials, or any derivative works thereof, from a Licensee as
      part of an integrated end user product, then Section 2 of this Agreement will not apply
      to you.

      iii. You must retain in all copies of the Llama Materials that you distribute the
      following attribution notice within a “Notice” text file distributed as a part of such
      copies: “Meta Llama 3 is licensed under the Meta Llama 3 Community License, Copyright ©
      Meta Platforms, Inc. All Rights Reserved.”

      iv. Your use of the Llama Materials must comply with applicable laws and regulations
      (including trade compliance laws and regulations) and adhere to the Acceptable Use Policy
      for the Llama Materials (available at https://llama.meta.com/llama3/use-policy), which
      is hereby incorporated by reference into this Agreement.

      v. You will not use the Llama Materials or any output or results of the Llama Materials
      to improve any other large language model (excluding Meta Llama 3 or derivative works
      thereof).

   2. Additional Commercial Terms.

   If, on the Meta Llama 3 version release date, the monthly active users of the products or
   services made available by or for Licensee, or Licensee’s affiliates, is greater than 700
   million monthly active users in the preceding calendar month, you must request a license
   from Meta, which Meta may grant to you in its sole discretion, and you are not authorized
   to exercise any of the rights under this Agreement unless or until Meta otherwise expressly
   grants you such rights.

   3. Disclaimer of Warranty.

   UNLESS REQUIRED BY APPLICABLE LAW, THE LLAMA MATERIALS AND ANY OUTPUT AND RESULTS THEREFROM
   ARE PROVIDED ON AN “AS IS” BASIS, WITHOUT WARRANTIES OF ANY KIND, AND META DISCLAIMS ALL
   WARRANTIES OF ANY KIND, BOTH EXPRESS AND IMPLIED, INCLUDING, WITHOUT LIMITATION, ANY
   WARRANTIES OF TITLE, NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
   YOU ARE SOLELY RESPONSIBLE FOR DETERMINING THE APPROPRIATENESS OF USING OR REDISTRIBUTING
   THE LLAMA MATERIALS AND ASSUME ANY RISKS ASSOCIATED WITH YOUR USE OF THE LLAMA MATERIALS
   AND ANY OUTPUT AND RESULTS.

   4. Limitation of Liability.

   IN NO EVENT WILL META OR ITS AFFILIATES BE LIABLE UNDER ANY THEORY OF LIABILITY, WHETHER IN
   CONTRACT, TORT, NEGLIGENCE, PRODUCTS LIABILITY, OR OTHERWISE, ARISING OUT OF THIS AGREEMENT,
   FOR ANY LOST PROFITS OR ANY INDIRECT, SPECIAL, CONSEQUENTIAL, INCIDENTAL, EXEMPLARY OR
   PUNITIVE DAMAGES, EVEN IF META OR ITS AFFILIATES HAVE BEEN ADVISED OF THE POSSIBILITY OF ANY
   OF THE FOREGOING.

   5. Intellectual Property.

   a. No trademark licenses are granted under this Agreement, and in connection with the Llama
   Materials, neither Meta nor Licensee may use any name or mark owned by or associated with
   the other or any of its affiliates, except as required for reasonable and customary use in
   describing and redistributing the Llama Materials or as set forth in this Section 5(a).
   Meta hereby grants you a license to use “Llama 3” (the “Mark”) solely as required to comply
   with the last sentence of Section 1.b.i. You will comply with Meta’s brand guidelines
   (currently accessible at https://about.meta.com/brand/resources/meta/company-brand/).
   All goodwill arising out of your use of the Mark will inure to the benefit of Meta.

   b. Subject to Meta’s ownership of Llama Materials and derivatives made by or for Meta, with
   respect to any derivative works and modifications of the Llama Materials that are made by
   you, as between you and Meta, you are and will be the owner of such derivative works and
   modifications.

   c. If you institute litigation or other proceedings against Meta or any entity (including a
   cross-claim or counterclaim in a lawsuit) alleging that the Llama Materials or Meta Llama 3
   outputs or results, or any portion of any of the foregoing, constitutes infringement of
   intellectual property or other rights owned or licensable by you, then any licenses granted
   to you under this Agreement shall terminate as of the date such litigation or claim is filed
   or instituted. You will indemnify and hold harmless Meta from and against any claim by any
   third party arising out of or related to your use or distribution of the Llama Materials.

   6. Term and Termination.

   The term of this Agreement will commence upon your acceptance of this Agreement or access
   to the Llama Materials and will continue in full force and effect until terminated in
   accordance with the terms and conditions herein. Meta may terminate this Agreement if you
   are in breach of any term or condition of this Agreement. Upon termination of this Agreement,
   you shall delete and cease use of the Llama Materials. Sections 3, 4, and 7 shall survive
   the termination of this Agreement.

   7. Governing Law and Jurisdiction.

   This Agreement will be governed and construed under the laws of the State of California
   without regard to choice of law principles, and the UN Convention on Contracts for the
   International Sale of Goods does not apply to this Agreement. The courts of California
   shall have exclusive jurisdiction of any dispute arising out of this Agreement.

   META LLAMA 3 ACCEPTABLE USE POLICY

   Meta is committed to promoting safe and fair use of its tools and features, including Meta
   Llama 3. If you access or use Meta Llama 3, you agree to this Acceptable Use Policy
   (“Policy”). The most recent copy of this policy can be found at
   https://llama.meta.com/llama3/use-policy.

   Prohibited Uses

   We want everyone to use Meta Llama 3 safely and responsibly. You agree you will not use, or
   allow others to use, Meta Llama 3 to:

   1. Violate the law or others’ rights, including to:

   a. Engage in, promote, generate, contribute to, encourage, plan, incite, or further illegal
   or unlawful activity or content, such as:

      i. Violence or terrorism
      ii. Exploitation or harm to children, including the solicitation, creation, acquisition,
      or dissemination of child exploitative content or failure to report Child Sexual Abuse
      Material
      iii. Human trafficking, exploitation, and sexual violence
      iv. The illegal distribution of information or materials to minors, including obscene
      materials, or failure to employ legally required age-gating in connection with such
      information or materials
      v. Sexual solicitation
      vi. Any other criminal activity

   b. Engage in, promote, incite, or facilitate the harassment, abuse, threatening, or
   bullying of individuals or groups of individuals

   c. Engage in, promote, incite, or facilitate discrimination or other unlawful or harmful
   conduct in the provision of employment, employment benefits, credit, housing, other economic
   benefits, or other essential goods and services

   d. Engage in the unauthorized or unlicensed practice of any profession including, but not
   limited to, financial, legal, medical/health, or related professional practices

   e. Collect, process, disclose, generate, or infer health, demographic, or other sensitive
   personal or private information about individuals without rights and consents required by
   applicable laws

   f. Engage in or facilitate any action or generate any content that infringes, misappropriates,
   or otherwise violates any third-party rights, including the outputs or results of any
   products or services using the Llama Materials

   g. Create, generate, or facilitate the creation of malicious code, malware, computer viruses
   or do anything else that could disable, overburden, interfere with or impair the proper
   working, integrity, operation, or appearance of a website or computer system

   2. Engage in, promote, incite, facilitate, or assist in the planning or development of
   activities that present a risk of death or bodily harm to individuals, including use of Meta
   Llama 3 related to the following:

   a. Military, warfare, nuclear industries or applications, espionage, use for materials or
   activities that are subject to the International Traffic Arms Regulations (ITAR) maintained
   by the United States Department of State
   b. Guns and illegal weapons (including weapon development)
   c. Illegal drugs and regulated/controlled substances
   d. Operation of critical infrastructure, transportation technologies, or heavy machinery
   e. Self-harm or harm to others, including suicide, cutting, and eating disorders
   f. Any content intended to incite or promote violence, abuse, or any infliction of bodily
   harm to an individual

   3. Intentionally deceive or mislead others, including use of Meta Llama 3 related to the
   following:

   a. Generating, promoting, or furthering fraud or the creation or promotion of disinformation
   b. Generating, promoting, or furthering defamatory content, including the creation of
   defamatory statements, images, or other content
   c. Generating, promoting, or further distributing spam
   d. Impersonating another individual without consent, authorization, or legal right
   e. Representing that the use of Meta Llama 3 or outputs are human-generated
   f. Generating or facilitating false online engagement, including fake reviews and other
   means of fake online engagement
   g. Fail to appropriately disclose to end users any known dangers of your AI system

   Please report any violation of this Policy, software “bug,” or other problems that could
   lead to a violation of this Policy through one of the following means:

   * Reporting issues with the model: https://github.com/meta-llama/llama3
   * Reporting risky content generated by the model: developers.facebook.com/llama_output_feedback
   * Reporting bugs and security concerns: facebook.com/whitehat/info
   * Reporting violations of the Acceptable Use Policy or unlicensed uses of Meta Llama 3:
      LlamaUseReport@meta.com

   ```

## Loguru - [MIT License](https://github.com/Delgan/loguru/blob/master/LICENSE)

   ```

   MIT License

   Copyright (c) 2017

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in all
   copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
   SOFTWARE.

   ```

## Mediapy - [Apache License 2.0](https://github.com/google/mediapy/blob/main/LICENSE)

   ```

                                    Apache License
                              Version 2.0, January 2004
                           http://www.apache.org/licenses/

      TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

      1. Definitions.

         "License" shall mean the terms and conditions for use, reproduction,
         and distribution as defined by Sections 1 through 9 of this document.

         "Licensor" shall mean the copyright owner or entity authorized by
         the copyright owner that is granting the License.

         "Legal Entity" shall mean the union of the acting entity and all
         other entities that control, are controlled by, or are under common
         control with that entity. For the purposes of this definition,
         "control" means (i) the power, direct or indirect, to cause the
         direction or management of such entity, whether by contract or
         otherwise, or (ii) ownership of fifty percent (50%) or more of the
         outstanding shares, or (iii) beneficial ownership of such entity.

         "You" (or "Your") shall mean an individual or Legal Entity
         exercising permissions granted by this License.

         "Source" form shall mean the preferred form for making modifications,
         including but not limited to software source code, documentation
         source, and configuration files.

         "Object" form shall mean any form resulting from mechanical
         transformation or translation of a Source form, including but
         not limited to compiled object code, generated documentation,
         and conversions to other media types.

         "Work" shall mean the work of authorship, whether in Source or
         Object form, made available under the License, as indicated by a
         copyright notice that is included in or attached to the work
         (an example is provided in the Appendix below).

         "Derivative Works" shall mean any work, whether in Source or Object
         form, that is based on (or derived from) the Work and for which the
         editorial revisions, annotations, elaborations, or other modifications
         represent, as a whole, an original work of authorship. For the purposes
         of this License, Derivative Works shall not include works that remain
         separable from, or merely link (or bind by name) to the interfaces of,
         the Work and Derivative Works thereof.

         "Contribution" shall mean any work of authorship, including
         the original version of the Work and any modifications or additions
         to that Work or Derivative Works thereof, that is intentionally
         submitted to Licensor for inclusion in the Work by the copyright owner
         or by an individual or Legal Entity authorized to submit on behalf of
         the copyright owner. For the purposes of this definition, "submitted"
         means any form of electronic, verbal, or written communication sent
         to the Licensor or its representatives, including but not limited to
         communication on electronic mailing lists, source code control systems,
         and issue tracking systems that are managed by, or on behalf of, the
         Licensor for the purpose of discussing and improving the Work, but
         excluding communication that is conspicuously marked or otherwise
         designated in writing by the copyright owner as "Not a Contribution."

         "Contributor" shall mean Licensor and any individual or Legal Entity
         on behalf of whom a Contribution has been received by Licensor and
         subsequently incorporated within the Work.

      2. Grant of Copyright License. Subject to the terms and conditions of
         this License, each Contributor hereby grants to You a perpetual,
         worldwide, non-exclusive, no-charge, royalty-free, irrevocable
         copyright license to reproduce, prepare Derivative Works of,
         publicly display, publicly perform, sublicense, and distribute the
         Work and such Derivative Works in Source or Object form.

      3. Grant of Patent License. Subject to the terms and conditions of
         this License, each Contributor hereby grants to You a perpetual,
         worldwide, non-exclusive, no-charge, royalty-free, irrevocable
         (except as stated in this section) patent license to make, have made,
         use, offer to sell, sell, import, and otherwise transfer the Work,
         where such license applies only to those patent claims licensable
         by such Contributor that are necessarily infringed by their
         Contribution(s) alone or by combination of their Contribution(s)
         with the Work to which such Contribution(s) was submitted. If You
         institute patent litigation against any entity (including a
         cross-claim or counterclaim in a lawsuit) alleging that the Work
         or a Contribution incorporated within the Work constitutes direct
         or contributory patent infringement, then any patent licenses
         granted to You under this License for that Work shall terminate
         as of the date such litigation is filed.

      4. Redistribution. You may reproduce and distribute copies of the
         Work or Derivative Works thereof in any medium, with or without
         modifications, and in Source or Object form, provided that You
         meet the following conditions:

         (a) You must give any other recipients of the Work or
             Derivative Works a copy of this License; and

         (b) You must cause any modified files to carry prominent notices
             stating that You changed the files; and

         (c) You must retain, in the Source form of any Derivative Works
             that You distribute, all copyright, patent, trademark, and
             attribution notices from the Source form of the Work,
             excluding those notices that do not pertain to any part of
             the Derivative Works; and

         (d) If the Work includes a "NOTICE" text file as part of its
             distribution, then any Derivative Works that You distribute must
             include a readable copy of the attribution notices contained
             within such NOTICE file, excluding those notices that do not
             pertain to any part of the Derivative Works, in at least one
             of the following places: within a NOTICE text file distributed
             as part of the Derivative Works; within the Source form or
             documentation, if provided along with the Derivative Works; or,
             within a display generated by the Derivative Works, if and
             wherever such third-party notices normally appear. The contents
             of the NOTICE file are for informational purposes only and
             do not modify the License. You may add Your own attribution
             notices within Derivative Works that You distribute, alongside
             or as an addendum to the NOTICE text from the Work, provided
             that such additional attribution notices cannot be construed
             as modifying the License.

         You may add Your own copyright statement to Your modifications and
         may provide additional or different license terms and conditions
         for use, reproduction, or distribution of Your modifications, or
         for any such Derivative Works as a whole, provided Your use,
         reproduction, and distribution of the Work otherwise complies with
         the conditions stated in this License.

      5. Submission of Contributions. Unless You explicitly state otherwise,
         any Contribution intentionally submitted for inclusion in the Work
         by You to the Licensor shall be under the terms and conditions of
         this License, without any additional terms or conditions.
         Notwithstanding the above, nothing herein shall supersede or modify
         the terms of any separate license agreement you may have executed
         with Licensor regarding such Contributions.

      6. Trademarks. This License does not grant permission to use the trade
         names, trademarks, service marks, or product names of the Licensor,
         except as required for reasonable and customary use in describing the
         origin of the Work and reproducing the content of the NOTICE file.

      7. Disclaimer of Warranty. Unless required by applicable law or
         agreed to in writing, Licensor provides the Work (and each
         Contributor provides its Contributions) on an "AS IS" BASIS,
         WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
         implied, including, without limitation, any warranties or conditions
         of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
         PARTICULAR PURPOSE. You are solely responsible for determining the
         appropriateness of using or redistributing the Work and assume any
         risks associated with Your exercise of permissions under this License.

      8. Limitation of Liability. In no event and under no legal theory,
         whether in tort (including negligence), contract, or otherwise,
         unless required by applicable law (such as deliberate and grossly
         negligent acts) or agreed to in writing, shall any Contributor be
         liable to You for damages, including any direct, indirect, special,
         incidental, or consequential damages of any character arising as a
         result of this License or out of the use or inability to use the
         Work (including but not limited to damages for loss of goodwill,
         work stoppage, computer failure or malfunction, or any and all
         other commercial damages or losses), even if such Contributor
         has been advised of the possibility of such damages.

      9. Accepting Warranty or Additional Liability. While redistributing
         the Work or Derivative Works thereof, You may choose to offer,
         and charge a fee for, acceptance of support, warranty, indemnity,
         or other liability obligations and/or rights consistent with this
         License. However, in accepting such obligations, You may act only
         on Your own behalf and on Your sole responsibility, not on behalf
         of any other Contributor, and only if You agree to indemnify,
         defend, and hold each Contributor harmless for any liability
         incurred by, or claims asserted against, such Contributor by reason
         of your accepting any such warranty or additional liability.

      END OF TERMS AND CONDITIONS

      APPENDIX: How to apply the Apache License to your work.

         To apply the Apache License to your work, attach the following
         boilerplate notice, with the fields enclosed by brackets "[]"
         replaced with your own identifying information. (Don't include
         the brackets!)  The text should be enclosed in the appropriate
         comment syntax for the file format. We also recommend that a
         file or class name and description of purpose be included on the
         same "printed page" as the copyright notice for easier
         identification within third-party archives.

      Copyright [yyyy] [name of copyright owner]

      Licensed under the Apache License, Version 2.0 (the "License");
      you may not use this file except in compliance with the License.
      You may obtain a copy of the License at

          http://www.apache.org/licenses/LICENSE-2.0

      Unless required by applicable law or agreed to in writing, software
      distributed under the License is distributed on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
      See the License for the specific language governing permissions and
      limitations under the License.

   ```

## Nltk - [Apache License 2.0](https://github.com/nltk/nltk/blob/develop/LICENSE.txt)

   ```

                                    Apache License
                              Version 2.0, January 2004
                           http://www.apache.org/licenses/

      TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

      1. Definitions.

         "License" shall mean the terms and conditions for use, reproduction,
         and distribution as defined by Sections 1 through 9 of this document.

         "Licensor" shall mean the copyright owner or entity authorized by
         the copyright owner that is granting the License.

         "Legal Entity" shall mean the union of the acting entity and all
         other entities that control, are controlled by, or are under common
         control with that entity. For the purposes of this definition,
         "control" means (i) the power, direct or indirect, to cause the
         direction or management of such entity, whether by contract or
         otherwise, or (ii) ownership of fifty percent (50%) or more of the
         outstanding shares, or (iii) beneficial ownership of such entity.

         "You" (or "Your") shall mean an individual or Legal Entity
         exercising permissions granted by this License.

         "Source" form shall mean the preferred form for making modifications,
         including but not limited to software source code, documentation
         source, and configuration files.

         "Object" form shall mean any form resulting from mechanical
         transformation or translation of a Source form, including but
         not limited to compiled object code, generated documentation,
         and conversions to other media types.

         "Work" shall mean the work of authorship, whether in Source or
         Object form, made available under the License, as indicated by a
         copyright notice that is included in or attached to the work
         (an example is provided in the Appendix below).

         "Derivative Works" shall mean any work, whether in Source or Object
         form, that is based on (or derived from) the Work and for which the
         editorial revisions, annotations, elaborations, or other modifications
         represent, as a whole, an original work of authorship. For the purposes
         of this License, Derivative Works shall not include works that remain
         separable from, or merely link (or bind by name) to the interfaces of,
         the Work and Derivative Works thereof.

         "Contribution" shall mean any work of authorship, including
         the original version of the Work and any modifications or additions
         to that Work or Derivative Works thereof, that is intentionally
         submitted to Licensor for inclusion in the Work by the copyright owner
         or by an individual or Legal Entity authorized to submit on behalf of
         the copyright owner. For the purposes of this definition, "submitted"
         means any form of electronic, verbal, or written communication sent
         to the Licensor or its representatives, including but not limited to
         communication on electronic mailing lists, source code control systems,
         and issue tracking systems that are managed by, or on behalf of, the
         Licensor for the purpose of discussing and improving the Work, but
         excluding communication that is conspicuously marked or otherwise
         designated in writing by the copyright owner as "Not a Contribution."

         "Contributor" shall mean Licensor and any individual or Legal Entity
         on behalf of whom a Contribution has been received by Licensor and
         subsequently incorporated within the Work.

      2. Grant of Copyright License. Subject to the terms and conditions of
         this License, each Contributor hereby grants to You a perpetual,
         worldwide, non-exclusive, no-charge, royalty-free, irrevocable
         copyright license to reproduce, prepare Derivative Works of,
         publicly display, publicly perform, sublicense, and distribute the
         Work and such Derivative Works in Source or Object form.

      3. Grant of Patent License. Subject to the terms and conditions of
         this License, each Contributor hereby grants to You a perpetual,
         worldwide, non-exclusive, no-charge, royalty-free, irrevocable
         (except as stated in this section) patent license to make, have made,
         use, offer to sell, sell, import, and otherwise transfer the Work,
         where such license applies only to those patent claims licensable
         by such Contributor that are necessarily infringed by their
         Contribution(s) alone or by combination of their Contribution(s)
         with the Work to which such Contribution(s) was submitted. If You
         institute patent litigation against any entity (including a
         cross-claim or counterclaim in a lawsuit) alleging that the Work
         or a Contribution incorporated within the Work constitutes direct
         or contributory patent infringement, then any patent licenses
         granted to You under this License for that Work shall terminate
         as of the date such litigation is filed.

      4. Redistribution. You may reproduce and distribute copies of the
         Work or Derivative Works thereof in any medium, with or without
         modifications, and in Source or Object form, provided that You
         meet the following conditions:

         (a) You must give any other recipients of the Work or
             Derivative Works a copy of this License; and

         (b) You must cause any modified files to carry prominent notices
             stating that You changed the files; and

         (c) You must retain, in the Source form of any Derivative Works
             that You distribute, all copyright, patent, trademark, and
             attribution notices from the Source form of the Work,
             excluding those notices that do not pertain to any part of
             the Derivative Works; and

         (d) If the Work includes a "NOTICE" text file as part of its
             distribution, then any Derivative Works that You distribute must
             include a readable copy of the attribution notices contained
             within such NOTICE file, excluding those notices that do not
             pertain to any part of the Derivative Works, in at least one
             of the following places: within a NOTICE text file distributed
             as part of the Derivative Works; within the Source form or
             documentation, if provided along with the Derivative Works; or,
             within a display generated by the Derivative Works, if and
             wherever such third-party notices normally appear. The contents
             of the NOTICE file are for informational purposes only and
             do not modify the License. You may add Your own attribution
             notices within Derivative Works that You distribute, alongside
             or as an addendum to the NOTICE text from the Work, provided
             that such additional attribution notices cannot be construed
             as modifying the License.

         You may add Your own copyright statement to Your modifications and
         may provide additional or different license terms and conditions
         for use, reproduction, or distribution of Your modifications, or
         for any such Derivative Works as a whole, provided Your use,
         reproduction, and distribution of the Work otherwise complies with
         the conditions stated in this License.

      5. Submission of Contributions. Unless You explicitly state otherwise,
         any Contribution intentionally submitted for inclusion in the Work
         by You to the Licensor shall be under the terms and conditions of
         this License, without any additional terms or conditions.
         Notwithstanding the above, nothing herein shall supersede or modify
         the terms of any separate license agreement you may have executed
         with Licensor regarding such Contributions.

      6. Trademarks. This License does not grant permission to use the trade
         names, trademarks, service marks, or product names of the Licensor,
         except as required for reasonable and customary use in describing the
         origin of the Work and reproducing the content of the NOTICE file.

      7. Disclaimer of Warranty. Unless required by applicable law or
         agreed to in writing, Licensor provides the Work (and each
         Contributor provides its Contributions) on an "AS IS" BASIS,
         WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
         implied, including, without limitation, any warranties or conditions
         of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
         PARTICULAR PURPOSE. You are solely responsible for determining the
         appropriateness of using or redistributing the Work and assume any
         risks associated with Your exercise of permissions under this License.

      8. Limitation of Liability. In no event and under no legal theory,
         whether in tort (including negligence), contract, or otherwise,
         unless required by applicable law (such as deliberate and grossly
         negligent acts) or agreed to in writing, shall any Contributor be
         liable to You for damages, including any direct, indirect, special,
         incidental, or consequential damages of any character arising as a
         result of this License or out of the use or inability to use the
         Work (including but not limited to damages for loss of goodwill,
         work stoppage, computer failure or malfunction, or any and all
         other commercial damages or losses), even if such Contributor
         has been advised of the possibility of such damages.

      9. Accepting Warranty or Additional Liability. While redistributing
         the Work or Derivative Works thereof, You may choose to offer,
         and charge a fee for, acceptance of support, warranty, indemnity,
         or other liability obligations and/or rights consistent with this
         License. However, in accepting such obligations, You may act only
         on Your own behalf and on Your sole responsibility, not on behalf
         of any other Contributor, and only if You agree to indemnify,
         defend, and hold each Contributor harmless for any liability
         incurred by, or claims asserted against, such Contributor by reason
         of your accepting any such warranty or additional liability.

      END OF TERMS AND CONDITIONS

      APPENDIX: How to apply the Apache License to your work.

         To apply the Apache License to your work, attach the following
         boilerplate notice, with the fields enclosed by brackets "[]"
         replaced with your own identifying information. (Don't include
         the brackets!)  The text should be enclosed in the appropriate
         comment syntax for the file format. We also recommend that a
         file or class name and description of purpose be included on the
         same "printed page" as the copyright notice for easier
         identification within third-party archives.

      Copyright [yyyy] [name of copyright owner]

      Licensed under the Apache License, Version 2.0 (the "License");
      you may not use this file except in compliance with the License.
      You may obtain a copy of the License at

          http://www.apache.org/licenses/LICENSE-2.0

      Unless required by applicable law or agreed to in writing, software
      distributed under the License is distributed on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
      See the License for the specific language governing permissions and
      limitations under the License.

   ```

## PEFT - [Apache License 2.0](https://github.com/huggingface/peft/blob/main/LICENSE)

   ```

                                    Apache License
                              Version 2.0, January 2004
                           http://www.apache.org/licenses/

      TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

      1. Definitions.

         "License" shall mean the terms and conditions for use, reproduction,
         and distribution as defined by Sections 1 through 9 of this document.

         "Licensor" shall mean the copyright owner or entity authorized by
         the copyright owner that is granting the License.

         "Legal Entity" shall mean the union of the acting entity and all
         other entities that control, are controlled by, or are under common
         control with that entity. For the purposes of this definition,
         "control" means (i) the power, direct or indirect, to cause the
         direction or management of such entity, whether by contract or
         otherwise, or (ii) ownership of fifty percent (50%) or more of the
         outstanding shares, or (iii) beneficial ownership of such entity.

         "You" (or "Your") shall mean an individual or Legal Entity
         exercising permissions granted by this License.

         "Source" form shall mean the preferred form for making modifications,
         including but not limited to software source code, documentation
         source, and configuration files.

         "Object" form shall mean any form resulting from mechanical
         transformation or translation of a Source form, including but
         not limited to compiled object code, generated documentation,
         and conversions to other media types.

         "Work" shall mean the work of authorship, whether in Source or
         Object form, made available under the License, as indicated by a
         copyright notice that is included in or attached to the work
         (an example is provided in the Appendix below).

         "Derivative Works" shall mean any work, whether in Source or Object
         form, that is based on (or derived from) the Work and for which the
         editorial revisions, annotations, elaborations, or other modifications
         represent, as a whole, an original work of authorship. For the purposes
         of this License, Derivative Works shall not include works that remain
         separable from, or merely link (or bind by name) to the interfaces of,
         the Work and Derivative Works thereof.

         "Contribution" shall mean any work of authorship, including
         the original version of the Work and any modifications or additions
         to that Work or Derivative Works thereof, that is intentionally
         submitted to Licensor for inclusion in the Work by the copyright owner
         or by an individual or Legal Entity authorized to submit on behalf of
         the copyright owner. For the purposes of this definition, "submitted"
         means any form of electronic, verbal, or written communication sent
         to the Licensor or its representatives, including but not limited to
         communication on electronic mailing lists, source code control systems,
         and issue tracking systems that are managed by, or on behalf of, the
         Licensor for the purpose of discussing and improving the Work, but
         excluding communication that is conspicuously marked or otherwise
         designated in writing by the copyright owner as "Not a Contribution."

         "Contributor" shall mean Licensor and any individual or Legal Entity
         on behalf of whom a Contribution has been received by Licensor and
         subsequently incorporated within the Work.

      2. Grant of Copyright License. Subject to the terms and conditions of
         this License, each Contributor hereby grants to You a perpetual,
         worldwide, non-exclusive, no-charge, royalty-free, irrevocable
         copyright license to reproduce, prepare Derivative Works of,
         publicly display, publicly perform, sublicense, and distribute the
         Work and such Derivative Works in Source or Object form.

      3. Grant of Patent License. Subject to the terms and conditions of
         this License, each Contributor hereby grants to You a perpetual,
         worldwide, non-exclusive, no-charge, royalty-free, irrevocable
         (except as stated in this section) patent license to make, have made,
         use, offer to sell, sell, import, and otherwise transfer the Work,
         where such license applies only to those patent claims licensable
         by such Contributor that are necessarily infringed by their
         Contribution(s) alone or by combination of their Contribution(s)
         with the Work to which such Contribution(s) was submitted. If You
         institute patent litigation against any entity (including a
         cross-claim or counterclaim in a lawsuit) alleging that the Work
         or a Contribution incorporated within the Work constitutes direct
         or contributory patent infringement, then any patent licenses
         granted to You under this License for that Work shall terminate
         as of the date such litigation is filed.

      4. Redistribution. You may reproduce and distribute copies of the
         Work or Derivative Works thereof in any medium, with or without
         modifications, and in Source or Object form, provided that You
         meet the following conditions:

         (a) You must give any other recipients of the Work or
             Derivative Works a copy of this License; and

         (b) You must cause any modified files to carry prominent notices
             stating that You changed the files; and

         (c) You must retain, in the Source form of any Derivative Works
             that You distribute, all copyright, patent, trademark, and
             attribution notices from the Source form of the Work,
             excluding those notices that do not pertain to any part of
             the Derivative Works; and

         (d) If the Work includes a "NOTICE" text file as part of its
             distribution, then any Derivative Works that You distribute must
             include a readable copy of the attribution notices contained
             within such NOTICE file, excluding those notices that do not
             pertain to any part of the Derivative Works, in at least one
             of the following places: within a NOTICE text file distributed
             as part of the Derivative Works; within the Source form or
             documentation, if provided along with the Derivative Works; or,
             within a display generated by the Derivative Works, if and
             wherever such third-party notices normally appear. The contents
             of the NOTICE file are for informational purposes only and
             do not modify the License. You may add Your own attribution
             notices within Derivative Works that You distribute, alongside
             or as an addendum to the NOTICE text from the Work, provided
             that such additional attribution notices cannot be construed
             as modifying the License.

         You may add Your own copyright statement to Your modifications and
         may provide additional or different license terms and conditions
         for use, reproduction, or distribution of Your modifications, or
         for any such Derivative Works as a whole, provided Your use,
         reproduction, and distribution of the Work otherwise complies with
         the conditions stated in this License.

      5. Submission of Contributions. Unless You explicitly state otherwise,
         any Contribution intentionally submitted for inclusion in the Work
         by You to the Licensor shall be under the terms and conditions of
         this License, without any additional terms or conditions.
         Notwithstanding the above, nothing herein shall supersede or modify
         the terms of any separate license agreement you may have executed
         with Licensor regarding such Contributions.

      6. Trademarks. This License does not grant permission to use the trade
         names, trademarks, service marks, or product names of the Licensor,
         except as required for reasonable and customary use in describing the
         origin of the Work and reproducing the content of the NOTICE file.

      7. Disclaimer of Warranty. Unless required by applicable law or
         agreed to in writing, Licensor provides the Work (and each
         Contributor provides its Contributions) on an "AS IS" BASIS,
         WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
         implied, including, without limitation, any warranties or conditions
         of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
         PARTICULAR PURPOSE. You are solely responsible for determining the
         appropriateness of using or redistributing the Work and assume any
         risks associated with Your exercise of permissions under this License.

      8. Limitation of Liability. In no event and under no legal theory,
         whether in tort (including negligence), contract, or otherwise,
         unless required by applicable law (such as deliberate and grossly
         negligent acts) or agreed to in writing, shall any Contributor be
         liable to You for damages, including any direct, indirect, special,
         incidental, or consequential damages of any character arising as a
         result of this License or out of the use or inability to use the
         Work (including but not limited to damages for loss of goodwill,
         work stoppage, computer failure or malfunction, or any and all
         other commercial damages or losses), even if such Contributor
         has been advised of the possibility of such damages.

      9. Accepting Warranty or Additional Liability. While redistributing
         the Work or Derivative Works thereof, You may choose to offer,
         and charge a fee for, acceptance of support, warranty, indemnity,
         or other liability obligations and/or rights consistent with this
         License. However, in accepting such obligations, You may act only
         on Your own behalf and on Your sole responsibility, not on behalf
         of any other Contributor, and only if You agree to indemnify,
         defend, and hold each Contributor harmless for any liability
         incurred by, or claims asserted against, such Contributor by reason
         of your accepting any such warranty or additional liability.

      END OF TERMS AND CONDITIONS

      APPENDIX: How to apply the Apache License to your work.

         To apply the Apache License to your work, attach the following
         boilerplate notice, with the fields enclosed by brackets "[]"
         replaced with your own identifying information. (Don't include
         the brackets!)  The text should be enclosed in the appropriate
         comment syntax for the file format. We also recommend that a
         file or class name and description of purpose be included on the
         same "printed page" as the copyright notice for easier
         identification within third-party archives.

      Copyright [yyyy] [name of copyright owner]

      Licensed under the Apache License, Version 2.0 (the "License");
      you may not use this file except in compliance with the License.
      You may obtain a copy of the License at

          http://www.apache.org/licenses/LICENSE-2.0

      Unless required by applicable law or agreed to in writing, software
      distributed under the License is distributed on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
      See the License for the specific language governing permissions and
      limitations under the License.

   ```

## Pillow - [MIT License](https://github.com/python-pillow/Pillow/blob/main/LICENSE)

   ```

   The Python Imaging Library (PIL) is

       Copyright © 1997-2011 by Secret Labs AB
       Copyright © 1995-2011 by Fredrik Lundh and contributors

   Pillow is the friendly PIL fork. It is

       Copyright © 2010 by Jeffrey A. Clark and contributors

   Like PIL, Pillow is licensed under the open source MIT-CMU License:

   By obtaining, using, and/or copying this software and/or its associated
   documentation, you agree that you have read, understood, and will comply
   with the following terms and conditions:

   Permission to use, copy, modify and distribute this software and its
   documentation for any purpose and without fee is hereby granted,
   provided that the above copyright notice appears in all copies, and that
   both that copyright notice and this permission notice appear in supporting
   documentation, and that the name of Secret Labs AB or the author not be
   used in advertising or publicity pertaining to distribution of the software
   without specific, written prior permission.

   SECRET LABS AB AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS
   SOFTWARE, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS.
   IN NO EVENT SHALL SECRET LABS AB OR THE AUTHOR BE LIABLE FOR ANY SPECIAL,
   INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
   LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
   OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
   PERFORMANCE OF THIS SOFTWARE.

   ```

## PyAV - [BSD 3-Clause "New" or "Revised" License](https://github.com/PyAV-Org/PyAV/blob/main/LICENSE.txt)

   ```

   Copyright retained by original committers. All rights reserved.

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are met:
       * Redistributions of source code must retain the above copyright
         notice, this list of conditions and the following disclaimer.
       * Redistributions in binary form must reproduce the above copyright
         notice, this list of conditions and the following disclaimer in the
         documentation and/or other materials provided with the distribution.
       * Neither the name of the project nor the names of its contributors may be
         used to endorse or promote products derived from this software without
         specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
   AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
   IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
   DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS BE LIABLE FOR ANY DIRECT,
   INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
   BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
   OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
   EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

   ```

## Pytorch_Retinaface - [MIT License](https://github.com/biubug6/Pytorch_Retinaface/blob/master/LICENSE.MIT)

   ```
   MIT License

   Copyright (c) 2019

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in all
   copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
   SOFTWARE.
   ```

## Sentencepiece - [Apache License 2.0](https://github.com/google/sentencepiece/blob/master/LICENSE)

   ```

                                    Apache License
                              Version 2.0, January 2004
                           http://www.apache.org/licenses/

      TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

      1. Definitions.

         "License" shall mean the terms and conditions for use, reproduction,
         and distribution as defined by Sections 1 through 9 of this document.

         "Licensor" shall mean the copyright owner or entity authorized by
         the copyright owner that is granting the License.

         "Legal Entity" shall mean the union of the acting entity and all
         other entities that control, are controlled by, or are under common
         control with that entity. For the purposes of this definition,
         "control" means (i) the power, direct or indirect, to cause the
         direction or management of such entity, whether by contract or
         otherwise, or (ii) ownership of fifty percent (50%) or more of the
         outstanding shares, or (iii) beneficial ownership of such entity.

         "You" (or "Your") shall mean an individual or Legal Entity
         exercising permissions granted by this License.

         "Source" form shall mean the preferred form for making modifications,
         including but not limited to software source code, documentation
         source, and configuration files.

         "Object" form shall mean any form resulting from mechanical
         transformation or translation of a Source form, including but
         not limited to compiled object code, generated documentation,
         and conversions to other media types.

         "Work" shall mean the work of authorship, whether in Source or
         Object form, made available under the License, as indicated by a
         copyright notice that is included in or attached to the work
         (an example is provided in the Appendix below).

         "Derivative Works" shall mean any work, whether in Source or Object
         form, that is based on (or derived from) the Work and for which the
         editorial revisions, annotations, elaborations, or other modifications
         represent, as a whole, an original work of authorship. For the purposes
         of this License, Derivative Works shall not include works that remain
         separable from, or merely link (or bind by name) to the interfaces of,
         the Work and Derivative Works thereof.

         "Contribution" shall mean any work of authorship, including
         the original version of the Work and any modifications or additions
         to that Work or Derivative Works thereof, that is intentionally
         submitted to Licensor for inclusion in the Work by the copyright owner
         or by an individual or Legal Entity authorized to submit on behalf of
         the copyright owner. For the purposes of this definition, "submitted"
         means any form of electronic, verbal, or written communication sent
         to the Licensor or its representatives, including but not limited to
         communication on electronic mailing lists, source code control systems,
         and issue tracking systems that are managed by, or on behalf of, the
         Licensor for the purpose of discussing and improving the Work, but
         excluding communication that is conspicuously marked or otherwise
         designated in writing by the copyright owner as "Not a Contribution."

         "Contributor" shall mean Licensor and any individual or Legal Entity
         on behalf of whom a Contribution has been received by Licensor and
         subsequently incorporated within the Work.

      2. Grant of Copyright License. Subject to the terms and conditions of
         this License, each Contributor hereby grants to You a perpetual,
         worldwide, non-exclusive, no-charge, royalty-free, irrevocable
         copyright license to reproduce, prepare Derivative Works of,
         publicly display, publicly perform, sublicense, and distribute the
         Work and such Derivative Works in Source or Object form.

      3. Grant of Patent License. Subject to the terms and conditions of
         this License, each Contributor hereby grants to You a perpetual,
         worldwide, non-exclusive, no-charge, royalty-free, irrevocable
         (except as stated in this section) patent license to make, have made,
         use, offer to sell, sell, import, and otherwise transfer the Work,
         where such license applies only to those patent claims licensable
         by such Contributor that are necessarily infringed by their
         Contribution(s) alone or by combination of their Contribution(s)
         with the Work to which such Contribution(s) was submitted. If You
         institute patent litigation against any entity (including a
         cross-claim or counterclaim in a lawsuit) alleging that the Work
         or a Contribution incorporated within the Work constitutes direct
         or contributory patent infringement, then any patent licenses
         granted to You under this License for that Work shall terminate
         as of the date such litigation is filed.

      4. Redistribution. You may reproduce and distribute copies of the
         Work or Derivative Works thereof in any medium, with or without
         modifications, and in Source or Object form, provided that You
         meet the following conditions:

         (a) You must give any other recipients of the Work or
             Derivative Works a copy of this License; and

         (b) You must cause any modified files to carry prominent notices
             stating that You changed the files; and

         (c) You must retain, in the Source form of any Derivative Works
             that You distribute, all copyright, patent, trademark, and
             attribution notices from the Source form of the Work,
             excluding those notices that do not pertain to any part of
             the Derivative Works; and

         (d) If the Work includes a "NOTICE" text file as part of its
             distribution, then any Derivative Works that You distribute must
             include a readable copy of the attribution notices contained
             within such NOTICE file, excluding those notices that do not
             pertain to any part of the Derivative Works, in at least one
             of the following places: within a NOTICE text file distributed
             as part of the Derivative Works; within the Source form or
             documentation, if provided along with the Derivative Works; or,
             within a display generated by the Derivative Works, if and
             wherever such third-party notices normally appear. The contents
             of the NOTICE file are for informational purposes only and
             do not modify the License. You may add Your own attribution
             notices within Derivative Works that You distribute, alongside
             or as an addendum to the NOTICE text from the Work, provided
             that such additional attribution notices cannot be construed
             as modifying the License.

         You may add Your own copyright statement to Your modifications and
         may provide additional or different license terms and conditions
         for use, reproduction, or distribution of Your modifications, or
         for any such Derivative Works as a whole, provided Your use,
         reproduction, and distribution of the Work otherwise complies with
         the conditions stated in this License.

      5. Submission of Contributions. Unless You explicitly state otherwise,
         any Contribution intentionally submitted for inclusion in the Work
         by You to the Licensor shall be under the terms and conditions of
         this License, without any additional terms or conditions.
         Notwithstanding the above, nothing herein shall supersede or modify
         the terms of any separate license agreement you may have executed
         with Licensor regarding such Contributions.

      6. Trademarks. This License does not grant permission to use the trade
         names, trademarks, service marks, or product names of the Licensor,
         except as required for reasonable and customary use in describing the
         origin of the Work and reproducing the content of the NOTICE file.

      7. Disclaimer of Warranty. Unless required by applicable law or
         agreed to in writing, Licensor provides the Work (and each
         Contributor provides its Contributions) on an "AS IS" BASIS,
         WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
         implied, including, without limitation, any warranties or conditions
         of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
         PARTICULAR PURPOSE. You are solely responsible for determining the
         appropriateness of using or redistributing the Work and assume any
         risks associated with Your exercise of permissions under this License.

      8. Limitation of Liability. In no event and under no legal theory,
         whether in tort (including negligence), contract, or otherwise,
         unless required by applicable law (such as deliberate and grossly
         negligent acts) or agreed to in writing, shall any Contributor be
         liable to You for damages, including any direct, indirect, special,
         incidental, or consequential damages of any character arising as a
         result of this License or out of the use or inability to use the
         Work (including but not limited to damages for loss of goodwill,
         work stoppage, computer failure or malfunction, or any and all
         other commercial damages or losses), even if such Contributor
         has been advised of the possibility of such damages.

      9. Accepting Warranty or Additional Liability. While redistributing
         the Work or Derivative Works thereof, You may choose to offer,
         and charge a fee for, acceptance of support, warranty, indemnity,
         or other liability obligations and/or rights consistent with this
         License. However, in accepting such obligations, You may act only
         on Your own behalf and on Your sole responsibility, not on behalf
         of any other Contributor, and only if You agree to indemnify,
         defend, and hold each Contributor harmless for any liability
         incurred by, or claims asserted against, such Contributor by reason
         of your accepting any such warranty or additional liability.

      END OF TERMS AND CONDITIONS

      APPENDIX: How to apply the Apache License to your work.

         To apply the Apache License to your work, attach the following
         boilerplate notice, with the fields enclosed by brackets "[]"
         replaced with your own identifying information. (Don't include
         the brackets!)  The text should be enclosed in the appropriate
         comment syntax for the file format. We also recommend that a
         file or class name and description of purpose be included on the
         same "printed page" as the copyright notice for easier
         identification within third-party archives.

      Copyright [yyyy] [name of copyright owner]

      Licensed under the Apache License, Version 2.0 (the "License");
      you may not use this file except in compliance with the License.
      You may obtain a copy of the License at

          http://www.apache.org/licenses/LICENSE-2.0

      Unless required by applicable law or agreed to in writing, software
      distributed under the License is distributed on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
      See the License for the specific language governing permissions and
      limitations under the License.

   ```

## Termcolor - [MIT License](https://github.com/termcolor/termcolor/blob/main/COPYING.txt)

   ```
   Copyright (c) 2008-2011 Volvox Development Team

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
   THE SOFTWARE.
   ```

## Transformers [Apache License 2.0](https://github.com/huggingface/transformers/blob/main/LICENSE)

   ```

   Copyright 2018- The Hugging Face team. All rights reserved.

                                    Apache License
                              Version 2.0, January 2004
                           http://www.apache.org/licenses/

      TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

      1. Definitions.

         "License" shall mean the terms and conditions for use, reproduction,
         and distribution as defined by Sections 1 through 9 of this document.

         "Licensor" shall mean the copyright owner or entity authorized by
         the copyright owner that is granting the License.

         "Legal Entity" shall mean the union of the acting entity and all
         other entities that control, are controlled by, or are under common
         control with that entity. For the purposes of this definition,
         "control" means (i) the power, direct or indirect, to cause the
         direction or management of such entity, whether by contract or
         otherwise, or (ii) ownership of fifty percent (50%) or more of the
         outstanding shares, or (iii) beneficial ownership of such entity.

         "You" (or "Your") shall mean an individual or Legal Entity
         exercising permissions granted by this License.

         "Source" form shall mean the preferred form for making modifications,
         including but not limited to software source code, documentation
         source, and configuration files.

         "Object" form shall mean any form resulting from mechanical
         transformation or translation of a Source form, including but
         not limited to compiled object code, generated documentation,
         and conversions to other media types.

         "Work" shall mean the work of authorship, whether in Source or
         Object form, made available under the License, as indicated by a
         copyright notice that is included in or attached to the work
         (an example is provided in the Appendix below).

         "Derivative Works" shall mean any work, whether in Source or Object
         form, that is based on (or derived from) the Work and for which the
         editorial revisions, annotations, elaborations, or other modifications
         represent, as a whole, an original work of authorship. For the purposes
         of this License, Derivative Works shall not include works that remain
         separable from, or merely link (or bind by name) to the interfaces of,
         the Work and Derivative Works thereof.

         "Contribution" shall mean any work of authorship, including
         the original version of the Work and any modifications or additions
         to that Work or Derivative Works thereof, that is intentionally
         submitted to Licensor for inclusion in the Work by the copyright owner
         or by an individual or Legal Entity authorized to submit on behalf of
         the copyright owner. For the purposes of this definition, "submitted"
         means any form of electronic, verbal, or written communication sent
         to the Licensor or its representatives, including but not limited to
         communication on electronic mailing lists, source code control systems,
         and issue tracking systems that are managed by, or on behalf of, the
         Licensor for the purpose of discussing and improving the Work, but
         excluding communication that is conspicuously marked or otherwise
         designated in writing by the copyright owner as "Not a Contribution."

         "Contributor" shall mean Licensor and any individual or Legal Entity
         on behalf of whom a Contribution has been received by Licensor and
         subsequently incorporated within the Work.

      2. Grant of Copyright License. Subject to the terms and conditions of
         this License, each Contributor hereby grants to You a perpetual,
         worldwide, non-exclusive, no-charge, royalty-free, irrevocable
         copyright license to reproduce, prepare Derivative Works of,
         publicly display, publicly perform, sublicense, and distribute the
         Work and such Derivative Works in Source or Object form.

      3. Grant of Patent License. Subject to the terms and conditions of
         this License, each Contributor hereby grants to You a perpetual,
         worldwide, non-exclusive, no-charge, royalty-free, irrevocable
         (except as stated in this section) patent license to make, have made,
         use, offer to sell, sell, import, and otherwise transfer the Work,
         where such license applies only to those patent claims licensable
         by such Contributor that are necessarily infringed by their
         Contribution(s) alone or by combination of their Contribution(s)
         with the Work to which such Contribution(s) was submitted. If You
         institute patent litigation against any entity (including a
         cross-claim or counterclaim in a lawsuit) alleging that the Work
         or a Contribution incorporated within the Work constitutes direct
         or contributory patent infringement, then any patent licenses
         granted to You under this License for that Work shall terminate
         as of the date such litigation is filed.

      4. Redistribution. You may reproduce and distribute copies of the
         Work or Derivative Works thereof in any medium, with or without
         modifications, and in Source or Object form, provided that You
         meet the following conditions:

         (a) You must give any other recipients of the Work or
             Derivative Works a copy of this License; and

         (b) You must cause any modified files to carry prominent notices
             stating that You changed the files; and

         (c) You must retain, in the Source form of any Derivative Works
             that You distribute, all copyright, patent, trademark, and
             attribution notices from the Source form of the Work,
             excluding those notices that do not pertain to any part of
             the Derivative Works; and

         (d) If the Work includes a "NOTICE" text file as part of its
             distribution, then any Derivative Works that You distribute must
             include a readable copy of the attribution notices contained
             within such NOTICE file, excluding those notices that do not
             pertain to any part of the Derivative Works, in at least one
             of the following places: within a NOTICE text file distributed
             as part of the Derivative Works; within the Source form or
             documentation, if provided along with the Derivative Works; or,
             within a display generated by the Derivative Works, if and
             wherever such third-party notices normally appear. The contents
             of the NOTICE file are for informational purposes only and
             do not modify the License. You may add Your own attribution
             notices within Derivative Works that You distribute, alongside
             or as an addendum to the NOTICE text from the Work, provided
             that such additional attribution notices cannot be construed
             as modifying the License.

         You may add Your own copyright statement to Your modifications and
         may provide additional or different license terms and conditions
         for use, reproduction, or distribution of Your modifications, or
         for any such Derivative Works as a whole, provided Your use,
         reproduction, and distribution of the Work otherwise complies with
         the conditions stated in this License.

      5. Submission of Contributions. Unless You explicitly state otherwise,
         any Contribution intentionally submitted for inclusion in the Work
         by You to the Licensor shall be under the terms and conditions of
         this License, without any additional terms or conditions.
         Notwithstanding the above, nothing herein shall supersede or modify
         the terms of any separate license agreement you may have executed
         with Licensor regarding such Contributions.

      6. Trademarks. This License does not grant permission to use the trade
         names, trademarks, service marks, or product names of the Licensor,
         except as required for reasonable and customary use in describing the
         origin of the Work and reproducing the content of the NOTICE file.

      7. Disclaimer of Warranty. Unless required by applicable law or
         agreed to in writing, Licensor provides the Work (and each
         Contributor provides its Contributions) on an "AS IS" BASIS,
         WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
         implied, including, without limitation, any warranties or conditions
         of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
         PARTICULAR PURPOSE. You are solely responsible for determining the
         appropriateness of using or redistributing the Work and assume any
         risks associated with Your exercise of permissions under this License.

      8. Limitation of Liability. In no event and under no legal theory,
         whether in tort (including negligence), contract, or otherwise,
         unless required by applicable law (such as deliberate and grossly
         negligent acts) or agreed to in writing, shall any Contributor be
         liable to You for damages, including any direct, indirect, special,
         incidental, or consequential damages of any character arising as a
         result of this License or out of the use or inability to use the
         Work (including but not limited to damages for loss of goodwill,
         work stoppage, computer failure or malfunction, or any and all
         other commercial damages or losses), even if such Contributor
         has been advised of the possibility of such damages.

      9. Accepting Warranty or Additional Liability. While redistributing
         the Work or Derivative Works thereof, You may choose to offer,
         and charge a fee for, acceptance of support, warranty, indemnity,
         or other liability obligations and/or rights consistent with this
         License. However, in accepting such obligations, You may act only
         on Your own behalf and on Your sole responsibility, not on behalf
         of any other Contributor, and only if You agree to indemnify,
         defend, and hold each Contributor harmless for any liability
         incurred by, or claims asserted against, such Contributor by reason
         of your accepting any such warranty or additional liability.

      END OF TERMS AND CONDITIONS

      APPENDIX: How to apply the Apache License to your work.

         To apply the Apache License to your work, attach the following
         boilerplate notice, with the fields enclosed by brackets "[]"
         replaced with your own identifying information. (Don't include
         the brackets!)  The text should be enclosed in the appropriate
         comment syntax for the file format. We also recommend that a
         file or class name and description of purpose be included on the
         same "printed page" as the copyright notice for easier
         identification within third-party archives.

      Copyright [yyyy] [name of copyright owner]

      Licensed under the Apache License, Version 2.0 (the "License");
      you may not use this file except in compliance with the License.
      You may obtain a copy of the License at

          http://www.apache.org/licenses/LICENSE-2.0

      Unless required by applicable law or agreed to in writing, software
      distributed under the License is distributed on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
      See the License for the specific language governing permissions and
      limitations under the License.

   ```
