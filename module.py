import torch
from torch.nn import CrossEntropyLoss 

from transformers import BertModel, BertForTokenClassification
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions, TokenClassifierOutput


class BertForTCwMRP(BertForTokenClassification):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModelForMRP(config, add_pooling_layer=False)

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        label_reps=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            label_reps=label_reps  # for masked token prediction
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class BertModelForMRP(BertModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        label_reps=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )

        embedding_output += label_reps  # masked labels

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )

#### ADDED PARTS ####

import torch.nn as nn
from transformers import BertPreTrainedModel

class BertForMultiTaskHSD(BertPreTrainedModel):
    """ BERT model for multi-task learning """
    
    def __init__(self, config, num_labels=3, num_target_groups=17):
        super().__init__(config)

        # Store task dimensions
        assert num_labels == 3 or num_labels == 2, "num_labels must be 2 or 3"
        self.num_labels = num_labels
        self.num_target_groups = num_target_groups
        
        # BERT encoder (shared between tasks)
        self.bert = BertModel(config)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Primary classification head for hate speech detection
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        
        # Secondary classification head for target group identification
        self.target_classifier = nn.Linear(config.hidden_size, num_target_groups)
        
        # Initialize weights
        self.init_weights()
        
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None, 
                labels=None, target_labels=None, output_attentions=None, 
                output_hidden_states=None, return_dict=None):
        """ Forward pass of the model."""
        
        # Default return_dict if not specified
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Pass through BERT encoder
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        
        # Get [CLS] token representation
        pooled_output = outputs[1]  # [batch_size, hidden_size]
        pooled_output = self.dropout(pooled_output)
        
        # Generate predictions for both tasks
        logits = self.classifier(pooled_output)  # [batch_size, num_labels]
        target_logits = self.target_classifier(pooled_output)  # [batch_size, num_target_groups]
        
        # Initialize loss to None
        total_loss = None
        
        # Calculate loss if labels are provided
        if labels is not None:

            # Calculate loss for hate speech detection
            loss_fct = nn.CrossEntropyLoss()
            classification_loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            # Calculate target loss if target_labels are provided
            if target_labels is not None:
                # Binary cross-entropy loss for target group identification
                target_loss_fct = nn.BCEWithLogitsLoss(reduction='none')
                target_loss = target_loss_fct(target_logits, target_labels.float())
                
                # Apply masking - only calculate target loss for hate/offensive samples
                if self.num_labels == 3:
                    is_hateful = (labels != 1).float().unsqueeze(1)  # [batch_size, 1]
                else:
                    is_hateful = (labels == 1).float().unsqueeze(1)  # [batch_size, 1]
                
                # Apply mask to loss
                masked_target_loss = (target_loss * is_hateful).mean()
                
                # Combine losses
                alpha = getattr(self.config, "alpha", 0.7)
                total_loss = alpha * classification_loss + (1 - alpha) * masked_target_loss
            else:
                total_loss = classification_loss
                
        # Prepare output
        if not return_dict:
            output = (logits, target_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output
        
        # Return dictionary with all outputs
        return {
            'loss': total_loss,
            'logits': logits,
            'target_logits': target_logits,
            'hidden_states': outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
            'attentions': outputs.attentions if hasattr(outputs, 'attentions') else None
        }
            
#### END OF ADDED PARTS ####