import jax
from jax import random, numpy as jnp
from flax.core.frozen_dict import freeze


def critic_update(rng, actor, critic, critic_target, temp, batch, discount):

    rng, key = random.split(rng)
    _, next_actions, next_log_probs, _ = actor.apply_fn(
        {"params": actor.params}, batch.next_images, batch.next_proprioceptions,
        False, key)  

    target_Q1, target_Q2 = critic_target.apply_fn(
        {"params": critic_target.params}, batch.next_images, 
        batch.next_proprioceptions, next_actions)
    
    temp_val = temp.apply_fn({"params": temp.params})
    target_V = jnp.minimum(target_Q1, target_Q2) - temp_val * next_log_probs

    target_Q = batch.rewards + batch.masks * discount * target_V

    def critic_loss_fn(critic_params):
        q1, q2 = critic.apply_fn( 
            {'params': critic_params}, batch.images, batch.proprioceptions, 
            batch.actions)
        
        critic_loss = ((q1 - target_Q)**2 + (q2 - target_Q)**2).mean()

        return critic_loss, {
            'critic_loss': critic_loss,
            'q1': q1.mean(),
            'q2': q2.mean()
        }

    grads, info = jax.grad(critic_loss_fn, has_aux=True)(critic.params)
    critic_new = critic.apply_gradients(grads=grads)

    return rng, critic_new, info


def actor_update(rng, actor, critic, temp, batch, use_critic_encoder=True):
    
    rng, key = random.split(rng)

    def actor_loss_fn(actor_params):    
        _, actions, log_probs, _ = actor.apply_fn(
            {"params": actor_params}, batch.images, batch.proprioceptions,
            False, key)

        q1, q2 = critic.apply_fn(
            {'params': critic.params}, batch.images, batch.proprioceptions, 
            actions)
        
        q = jnp.minimum(q1, q2)
        temp_val = temp.apply_fn({"params": temp.params})
        actor_loss = (log_probs * temp_val - q).mean()

        return actor_loss, {
            'actor_loss': actor_loss,
            'entropy': -log_probs.mean()
        }
    
    if use_critic_encoder:
        params = actor.params
        params['encoder'] = critic.params['encoder']
        actor = actor.replace(params=params)

    grads, info = jax.grad(actor_loss_fn, has_aux=True)(actor.params)
    actor_new = actor.apply_gradients(grads=grads)

    return rng, actor_new, info

def temp_update(temp, entropy, target_entropy):

    def temperature_loss_fn(temp_params):
        temperature = temp.apply_fn({'params': temp_params})
        temp_loss = temperature * (entropy - target_entropy).mean()
        return temp_loss, {
            'temperature': temperature, 
            'temp_loss': temp_loss}

    grads, info = jax.grad(temperature_loss_fn, has_aux=True)(temp.params)
    temp_new = temp.apply_gradients(grads=grads)

    return temp_new, info

def target_update(critic, critic_target, tau):
    new_target_params = jax.tree_map(
        lambda p, tp: p * tau + tp * (1 - tau), critic.params,
        critic_target.params)

    return critic_target.replace(params=new_target_params)