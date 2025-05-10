label_link_meld = '/root/autodl-tmp/dataset/meld/train_sent_emo.csv'
label_meld = pd.read_csv(label_link_meld)
for i in range(len(label_meld)):
    dialogue_id = label_meld.loc[i, 'Dialogue_ID']
    utterance_id = label_meld.loc[i, 'Utterance_ID']
    clip_id = f'dia{dialogue_id}_utt{utterance_id}.jpg'
    frame_path = osp.join(self.data_dir, f'{self.split}_splits', clip_id)