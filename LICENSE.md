# ZEN-LipSync ライセンス 1.0 ／ ZEN-LipSync License 1.0

Copyright (c) 2026 Zenryoku Kikai Inc. (全力機械株式会社) — https://zenryokukikai.com

## 日本語

### 1. 対象

このライセンスは、次のもの（以下「本ソフトウェア」）に適用されます。どちらのリポジトリにも、同じ文面のこのライセンスが入っています。
- ZEN-LipSync の SDK（リポジトリ zenryokukikai/zen-lipsync、npm パッケージ @zenryokukikai/zen-lipsync）のソースコードと文書
- ZEN-LipSync のアバター（リポジトリ zenryokukikai/zen-lipsync-avatars）のモデルの重み、アバター「全力 マリン」「全力 ハシロウ」「全力 キッカ」の素材（`avatars/marine/`・`avatars/hashiro/`・`avatars/kikka/` の待機映像・口元の画像・特徴データなど。以下「アバター素材」）、デモ、使い方の例、文書

THIRD_PARTY_NOTICES.md に挙げた第三者の部分は、それぞれのライセンスに従います。
アバター素材には、2 章・3 章に加えて、4 章の条件も適用されます。

### 2. 許可すること

次の 3 章の条件（1〜5）を守る限り、誰でも、無料で、次のことができます。
- 本ソフトウェアを使う。
- 本ソフトウェアを自分のアプリ・Web サイト・サービス・作品の中で使う。
- 本ソフトウェアを（改変せずに）自分のソフトウェアに組み込む。
- 本ソフトウェアで作った映像を使う（配信・販売・作品への利用を含む）。

これらは、商用・非商用を問いません。アバター素材は、4 章の範囲で使えます。

### 3. 条件

1. **クレジットの表示（必須）**
   本ソフトウェア、またはそれで作った映像を使うアプリ・Web サイト・サービス・作品では、利用者が普通に見つけられる場所に、次のクレジットを表示してください。クレジット中の URL はリンクにしてください（リンクを埋め込めない媒体では URL を文字で書けば足ります）。
   - クレジット：「リップシンク：ZEN-LipSync（https://zenryokukikai.com ）」、または英語の「Lip-sync: ZEN-LipSync (https://zenryokukikai.com )」
   - ZEN-TTS と併用する場合は、1 行にまとめて構いません（例：「音声合成：ZEN-TTS / リップシンク：ZEN-LipSync（https://zenryokukikai.com ）」）。
   - 表示する場所の例：画面、クレジット、「このアプリについて」のページ、説明欄、概要欄
   - 動画だけの作品：説明欄や概要欄など、作品と一緒に示される場所に表示してください。

2. **改変の禁止**
   本ソフトウェア（ソースコード、モデルの重み、アバター素材を含む）を改変してはいけません。自分のアプリのビルドの中で、改変せずに依存として取り込み、バンドル・圧縮（minify）することは改変にあたりません。

3. **再配布の禁止**
   本ソフトウェア、その重み、アバター素材を、第三者に再配布・再許諾・販売・譲渡・貸与してはいけません。自分のアプリ・サービスの中で使うことはできますが、モデルや重みを第三者に渡すことはできません。

4. **使ってはいけない目的**
   本ソフトウェアを、次の目的に使ってはいけません。
   - 実在する人になりすますこと
   - 本人の同意なく、実在する人が話しているように見せること
   - 人物が実際には言っていないことを、言ったかのように見せて、人をだますこと
   - 人の名誉や信用を傷つけること
   - 法令に違反する目的
   - 他人の権利（肖像権、パブリシティ権を含む）を侵害する目的
   - 詐欺の目的

5. **学習・蒸留・評価への利用の禁止**
   本ソフトウェア、その重み、および本ソフトウェアで作った映像を、他の機械学習モデルの学習、蒸留、評価用のデータとして使ってはいけません。

### 4. アバター素材（「全力 マリン」「全力 ハシロウ」「全力 キッカ」）

アバター「全力 マリン」「全力 ハシロウ」「全力 キッカ」は、全力機械株式会社のアバターです。アバター素材（`avatars/marine/`、`avatars/hashiro/`、`avatars/kikka/` の中身）は、次の範囲で使えます。同じ条件が、3 人のどれにも当てはまります。

1. **使えること**
   - 本ソフトウェアでリップシンクさせたアバターとして、アプリ・Web サイト・サービス・作品の中で使うこと（商用を含む）。たとえば、案内役、受付、学習の講師、デモ。
   - そのとき、アバターが「全力 マリン」「全力 ハシロウ」「全力 キッカ」のどれかというアバター（AI による映像）であることが分かるようにしてください。

2. **使ってはいけないこと**
   アバター素材を、次のことに使ってはいけません。
   - 実在する人のふりをさせること。見る人に、実在する人が本当に話していると思わせること。
   - アダルト、性的なもの、暴力的なものなど、公序良俗に反する内容
   - 人や団体の名誉や信用を傷つける内容、差別的な内容
   - 政治的な広告や選挙運動、宗教の勧誘
   - 詐欺、なりすまし、誤った情報を広めること
   - 法令に違反すること
   - アバターの顔や映像を取り出して、ほかの顔・映像の作成や、ほかのモデルの学習に使うこと
   - 「全力 マリン」「全力 ハシロウ」「全力 キッカ」の名前や姿を、全力機械株式会社の公式なものと誤解されるように使うこと

3. 全力機械株式会社は、アバター素材の使い方がこの章に反すると判断した場合、その使用の中止を求めることができます。

### 5. 保証しないこと

本ソフトウェアは「現状のまま」提供します。全力機械株式会社は、次のことを保証しません。
- 品質が正しいこと
- 特定の目的に使えること
- 第三者の権利を侵害しないこと

本ソフトウェアの使用によって生じた損害について、法令で許される範囲で、全力機械株式会社は責任を負いません。

### 6. ライセンスの終了

3 章または 4 章の条件に違反した場合、このライセンスで与えた許可は自動的に終わります。
ただし、違反に気づいてから 30 日以内に違反をやめた場合は、許可は元に戻ります。

### 7. 準拠法

このライセンスは、日本法に従って解釈します。

## English (translation; the Japanese text prevails)

### 1. Scope
This license covers (the "Software"; both repositories carry this same text): the source code and documents of the ZEN-LipSync SDK (repository zenryokukikai/zen-lipsync, npm package @zenryokukikai/zen-lipsync); and, in the repository zenryokukikai/zen-lipsync-avatars, the model weights, the materials of the avatars "Zenryoku Marine", "Zenryoku Hashiro" and "Zenryoku Kikka" (the idle videos, mouth images, features and other data in `avatars/marine/`, `avatars/hashiro/` and `avatars/kikka/`, the "Avatar Materials"), the demo, the examples and the documents. Third-party parts listed in THIRD_PARTY_NOTICES.md are under their own licenses. The Avatar Materials are also subject to Section 4.

### 2. Grant
Anyone may, free of charge and for commercial or non-commercial purposes:
- use the Software inside your own app, website, service or work, and embed it, unmodified, in your own software;
- use video produced with the Software (including distribution, sale and use in a work),

subject to the conditions in Section 3. The Avatar Materials may be used within Section 4.

### 3. Conditions
1. **Credit (required).** Any app, website, service or work that uses the Software or video produced with it must show, in a place where its users can reasonably find it, the credit "Lip-sync: ZEN-LipSync (https://zenryokukikai.com)" (or the Japanese equivalent), with the URL as a hyperlink. Where links cannot be embedded, the URL written as text is enough. When used together with ZEN-TTS, one combined line is fine (e.g. "Speech synthesis: ZEN-TTS / Lip-sync: ZEN-LipSync (https://zenryokukikai.com)").

   Suitable places include the UI, the credits, an about page or the description. For video-only works, use the accompanying description.
2. **No modification.** Do not modify the Software (including its source code, model weights and Avatar Materials). Including it unmodified as a dependency in your own build, bundled or minified, is not a modification.
3. **No redistribution.** Do not redistribute, sublicense, sell, transfer or lend the Software, its weights or the Avatar Materials to a third party. Using them inside your own app or service is allowed; handing the model or weights to someone else is not.
4. **Prohibited uses.** Do not use the Software to:
   - impersonate a real person;
   - make a real person appear to speak without that person's consent;
   - deceive people by showing a person saying something they did not say;
   - damage anyone's reputation or credit;
   - break the law;
   - infringe others' rights, including portrait and publicity rights;
   - commit fraud.
5. **No use for training, distillation or evaluation.** Do not use the Software, its weights, or video produced with the Software as data to train, distil or evaluate other machine-learning models.

### 4. The Avatar Materials ("Zenryoku Marine", "Zenryoku Hashiro", "Zenryoku Kikka")
"Zenryoku Marine", "Zenryoku Hashiro" and "Zenryoku Kikka" are avatars of Zenryoku Kikai Inc. The same conditions apply to each of the three.

1. **Allowed.** Use the Avatar Materials as the avatar lip-synced with the Software in apps, websites, services and works, including commercial ones (for example as a guide, a receptionist, a teacher or in demos). Make it clear that the avatar is one of "Zenryoku Marine", "Zenryoku Hashiro" and "Zenryoku Kikka", an avatar (AI-generated video).
2. **Prohibited.** Do not use the Avatar Materials:
   - to pass the avatar off as a real person, or to make viewers believe a real person is actually speaking;
   - in adult, sexual, violent or otherwise offensive content;
   - in content that damages the reputation or credit of any person or organisation, or that is discriminatory;
   - in political advertising, election campaigns or religious solicitation;
   - for fraud, impersonation or spreading false information;
   - to break the law;
   - to extract the face or video for making other faces or videos, or for training other models;
   - in a way that suggests official endorsement by Zenryoku Kikai Inc.
3. Zenryoku Kikai Inc. may ask you to stop a use of the Avatar Materials that it considers contrary to this Section.

### 5. No warranty
The Software is provided "as is". There is no warranty of any kind, including fitness for a particular purpose or non-infringement. To the extent permitted by law, Zenryoku Kikai Inc. is not liable for any damages arising from its use.

### 6. Termination
The grant ends automatically if Section 3 or 4 is breached. It is reinstated if the breach is cured within 30 days of becoming aware of it.

### 7. Governing law
This license is governed by the laws of Japan.
