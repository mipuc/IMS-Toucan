# Copyright 2020 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
# Adapted by Florian Lux 2021

import torch
import torch.nn.functional as F

from Layers.Attention import GuidedAttentionLoss
from Layers.RNNAttention import AttForward
from Layers.RNNAttention import AttForwardTA
from Layers.RNNAttention import AttLoc
from Layers.TacotronDecoder import Decoder
from Layers.TacotronEncoder import Encoder
from TrainingInterfaces.Text_to_Spectrogram.Tacotron2.AlignmentLoss import AlignmentLoss
from TrainingInterfaces.Text_to_Spectrogram.Tacotron2.Tacotron2Loss import Tacotron2Loss
from Utility.SoftDTW.sdtw_cuda_loss import SoftDTW
from Utility.utils import make_pad_mask


class Tacotron2(torch.nn.Module):
    """
    Tacotron2 module.

    This is a module of Spectrogram prediction network in Tacotron2

    .. _`Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions`:
       https://arxiv.org/abs/1712.05884
   """

    def __init__(
            self,
            # network structure related
            idim,
            odim,
            embed_dim=512,
            elayers=1,
            eunits=512,
            econv_layers=3,
            econv_chans=512,
            econv_filts=5,
            atype="forward_ta",
            adim=512,
            aconv_chans=32,
            aconv_filts=15,
            cumulate_att_w=True,
            dlayers=2,
            dunits=1024,
            prenet_layers=2,
            prenet_units=256,
            postnet_layers=5,
            postnet_chans=512,
            postnet_filts=5,
            output_activation=None,
            use_batch_norm=True,
            use_concate=True,
            use_residual=False,
            reduction_factor=1,
            spk_embed_dim=None,
            # training related
            dropout_rate=0.5,
            zoneout_rate=0.1,
            use_masking=False,
            use_weighted_masking=True,
            bce_pos_weight=10.0,
            loss_type="L1+L2",
            use_guided_attn_loss=True,
            guided_attn_loss_sigma=0.4,
            guided_attn_loss_lambda=1.0,
            use_dtw_loss=False,
            use_alignment_loss=True,
            speaker_embedding_projection_size=64):
        super().__init__()

        # store hyperparameters
        self.use_dtw_loss = use_dtw_loss
        self.use_alignment_loss = use_alignment_loss
        self.idim = idim
        self.odim = odim
        self.eos = idim - 1
        self.spk_embed_dim = spk_embed_dim
        print(spk_embed_dim)
        self.cumulate_att_w = cumulate_att_w
        self.reduction_factor = reduction_factor
        self.use_guided_attn_loss = use_guided_attn_loss
        self.loss_type = loss_type

        # define activation function for the final output
        if output_activation is None:
            self.output_activation_fn = None
        elif hasattr(F, output_activation):
            self.output_activation_fn = getattr(F, output_activation)
        else:
            raise ValueError(f"there is no such an activation function. " f"({output_activation})")

        # set padding idx
        padding_idx = 0
        self.padding_idx = padding_idx

        # define network modules
        self.enc = Encoder(idim=idim,
                           embed_dim=embed_dim,
                           elayers=elayers,
                           eunits=eunits,
                           econv_layers=econv_layers,
                           econv_chans=econv_chans,
                           econv_filts=econv_filts,
                           use_batch_norm=use_batch_norm,
                           use_residual=use_residual,
                           dropout_rate=dropout_rate,
                           padding_idx=padding_idx, )

        if spk_embed_dim is not None:
            self.encoder_speakerembedding_projection = torch.nn.Linear(eunits + speaker_embedding_projection_size, eunits)
            # embedding projection derived from https://arxiv.org/pdf/1705.08947.pdf
            self.embedding_projection = torch.nn.Sequential(torch.nn.Linear(spk_embed_dim, speaker_embedding_projection_size),
                                                            torch.nn.Softsign())
        else:
            speaker_embedding_projection_size = None
        dec_idim = eunits

        if atype == "location":
            att = AttLoc(dec_idim, dunits, adim, aconv_chans, aconv_filts)
        elif atype == "forward":
            att = AttForward(dec_idim, dunits, adim, aconv_chans, aconv_filts)
            if self.cumulate_att_w:
                self.cumulate_att_w = False
        elif atype == "forward_ta":
            att = AttForwardTA(dec_idim, dunits, adim, aconv_chans, aconv_filts, odim)
            if self.cumulate_att_w:
                self.cumulate_att_w = False
        else:
            raise NotImplementedError("Support only location or forward")
        self.dec = Decoder(idim=dec_idim,
                           odim=odim,
                           att=att,
                           dlayers=dlayers,
                           dunits=dunits,
                           prenet_layers=prenet_layers,
                           prenet_units=prenet_units,
                           postnet_layers=postnet_layers,
                           postnet_chans=postnet_chans,
                           postnet_filts=postnet_filts,
                           output_activation_fn=self.output_activation_fn,
                           cumulate_att_w=self.cumulate_att_w,
                           use_batch_norm=use_batch_norm,
                           use_concate=use_concate,
                           dropout_rate=dropout_rate,
                           zoneout_rate=zoneout_rate,
                           reduction_factor=reduction_factor,
                           speaker_embedding_projection_size=speaker_embedding_projection_size)
        self.taco2_loss = Tacotron2Loss(use_masking=use_masking,
                                        use_weighted_masking=use_weighted_masking,
                                        bce_pos_weight=bce_pos_weight, )
        if self.use_guided_attn_loss:
            self.guided_att_loss_start = GuidedAttentionLoss(sigma=guided_attn_loss_sigma,
                                                             alpha=guided_attn_loss_lambda * 10, )
            self.guided_att_loss_final = GuidedAttentionLoss(sigma=guided_attn_loss_sigma,
                                                             alpha=guided_attn_loss_lambda, )
        if self.use_dtw_loss:
            self.dtw_criterion = SoftDTW(use_cuda=True, gamma=0.1)

        if self.use_alignment_loss:
            self.alignment_loss = AlignmentLoss()

    def forward(self,
                text,
                text_lengths,
                speech,
                speech_lengths,
                step,
                speaker_embeddings=None,
                return_mels=False):
        """
        Calculate forward propagation.

        Args:
            step: current number of update steps taken as indicator when to start binarizing
            text (LongTensor): Batch of padded character ids (B, Tmax).
            text_lengths (LongTensor): Batch of lengths of each input batch (B,).
            speech (Tensor): Batch of padded target features (B, Lmax, odim).
            speech_lengths (LongTensor): Batch of the lengths of each target (B,).
            speaker_embeddings (Tensor, optional): Batch of speaker embeddings (B, spk_embed_dim).

        Returns:
            Tensor: Loss scalar value.
            Dict: Statistics to be monitored.
            Tensor: Weight value.
        """
        text = text[:, : text_lengths.max()]  # for data-parallel
        speech = speech[:, : speech_lengths.max()]  # for data-parallel

        # Add eos at the last of sequence
        xs = F.pad(text, [0, 1], "constant", self.padding_idx)
        for i, l in enumerate(text_lengths):
            xs[i, l] = self.eos
        ilens = text_lengths + 1

        ys = speech
        olens = speech_lengths

        # make labels for stop prediction
        labels = make_pad_mask(olens - 1).to(ys.device, ys.dtype)
        labels = F.pad(labels, [0, 1], "constant", 1.0)

        # calculate tacotron2 outputs
        after_outs, before_outs, logits, att_ws = self._forward(xs, ilens, ys, olens, speaker_embeddings)

        # modify mod part of groundtruth
        if self.reduction_factor > 1:
            assert olens.ge(self.reduction_factor).all(), "Output length must be greater than or equal to reduction factor."
            olens = olens.new([olen - olen % self.reduction_factor for olen in olens])
            max_out = max(olens)
            ys = ys[:, :max_out]
            labels = labels[:, :max_out]
            labels = torch.scatter(labels, 1, (olens - 1).unsqueeze(1), 1.0)  # see #3388

        # calculate taco2 loss
        l1_loss, mse_loss, bce_loss = self.taco2_loss(after_outs, before_outs, logits, ys, labels, olens)
        if self.loss_type == "L1+L2":
            loss = l1_loss + mse_loss + bce_loss
        elif self.loss_type == "L1":
            loss = l1_loss + bce_loss
        elif self.loss_type == "L2":
            loss = mse_loss + bce_loss
        else:
            raise ValueError(f"unknown --loss-type {self.loss_type}")

        # calculate dtw loss
        if self.use_dtw_loss:
            dtw_loss = self.dtw_criterion(after_outs, speech).mean() / 2000.0  # division to balance orders of magnitude
            loss += dtw_loss

        # calculate attention loss
        if self.use_guided_attn_loss:
            if self.reduction_factor > 1:
                olens_in = olens.new([olen // self.reduction_factor for olen in olens])
            else:
                olens_in = olens
            if step < 500:
                attn_loss = self.guided_att_loss_start(att_ws, ilens, olens_in)
                # build a prior in the attention map for the forward algorithm to take over
            else:
                attn_loss = self.guided_att_loss_final(att_ws, ilens, olens_in)
            loss = loss + attn_loss

        # calculate alignment loss
        if self.use_alignment_loss:
            if self.reduction_factor > 1:
                olens_in = olens.new([olen // self.reduction_factor for olen in olens])
            else:
                olens_in = olens
            align_loss = self.alignment_loss(att_ws, ilens, olens_in, step)
            loss = loss + align_loss
        if return_mels:
            return loss, after_outs
        return loss

    def _forward(self,
                 xs,
                 ilens,
                 ys,
                 olens,
                 speaker_embeddings):
        hs, hlens = self.enc(xs, ilens)
        if speaker_embeddings is not None:
            projected_speaker_embeddings = self.embedding_projection(speaker_embeddings)
        else:
            projected_speaker_embeddings = None
        if self.spk_embed_dim is not None:
            hs = self._integrate_with_spk_embed(hs, projected_speaker_embeddings)
        return self.dec(hs, hlens, ys, projected_speaker_embeddings)

    def inference(self,
                  text: torch.Tensor,
                  speech: torch.Tensor = None,
                  speaker_embeddings: torch.Tensor = None,
                  threshold: float = 0.5,
                  minlenratio: float = 0.0,
                  maxlenratio: float = 10.0,
                  use_att_constraint: bool = False,
                  backward_window: int = 1,
                  forward_window: int = 3,
                  use_teacher_forcing: bool = False, ):
        """
        Generate the sequence of features given the sequences of characters.

        Args:
            text (LongTensor): Input sequence of characters (T,).
            speech (Tensor, optional): Feature sequence to extract style (N, idim).
            speaker_embeddings (Tensor, optional): Speaker embedding vector (spk_embed_dim,).
            threshold (float, optional): Threshold in inference.
            minlenratio (float, optional): Minimum length ratio in inference.
            maxlenratio (float, optional): Maximum length ratio in inference.
            use_att_constraint (bool, optional): Whether to apply attention constraint.
            backward_window (int, optional): Backward window in attention constraint.
            forward_window (int, optional): Forward window in attention constraint.
            use_teacher_forcing (bool, optional): Whether to use teacher forcing.

        Returns:
            Tensor: Output sequence of features (L, odim).
            Tensor: Output sequence of stop probabilities (L,).
            Tensor: Attention weights (L, T).
        """
        x = text
        y = speech
        speaker_embedding = speaker_embeddings

        # add eos at the last of sequence
        x = F.pad(x, [0, 1], "constant", self.eos)

        # inference with teacher forcing
        if use_teacher_forcing:
            assert speech is not None, "speech must be provided with teacher forcing."

            xs, ys = x.unsqueeze(0), y.unsqueeze(0)
            speaker_embeddings = None if speaker_embedding is None else speaker_embedding.unsqueeze(0)
            ilens = x.new_tensor([xs.size(1)]).long()
            olens = y.new_tensor([ys.size(1)]).long()
            outs, _, _, att_ws = self._forward(xs, ilens, ys, olens, speaker_embeddings)

            return outs[0], None, att_ws[0]

        # inference
        h = self.enc.inference(x)
        if self.spk_embed_dim is not None:
            projected_speaker_embedding = self.embedding_projection(speaker_embedding)
            hs, speaker_embeddings = h.unsqueeze(0), projected_speaker_embedding.unsqueeze(0)
            h = self._integrate_with_spk_embed(hs, speaker_embeddings)[0]
        else:
            speaker_embeddings = None
        outs, probs, att_ws = self.dec.inference(h,
                                                 threshold=threshold,
                                                 minlenratio=minlenratio,
                                                 maxlenratio=maxlenratio,
                                                 use_att_constraint=use_att_constraint,
                                                 backward_window=backward_window,
                                                 forward_window=forward_window,
                                                 speaker_embedding=speaker_embeddings)

        return outs, probs, att_ws

    def _integrate_with_spk_embed(self, hs, speaker_embeddings):
        """
        Integrate speaker embedding with hidden states.

        Args:
            hs (Tensor): Batch of hidden state sequences (B, Tmax, adim).
            speaker_embeddings (Tensor): Batch of speaker embeddings (B, spk_embed_dim).

        Returns:
            Tensor: Batch of integrated hidden state sequences (B, Tmax, adim).

        """
        # concat hidden states with spk embeds and then apply projection
        speaker_embeddings_expanded = F.normalize(speaker_embeddings).unsqueeze(1).expand(-1, hs.size(1), -1)
        hs = self.encoder_speakerembedding_projection(torch.cat([hs, speaker_embeddings_expanded], dim=-1))
        return hs
