import jax
from jax import random, numpy as jnp
import flax


def critic_update(rng, 
                  actor, 
                  critic, 
                  critic_target_params, 
                  temp, 
                  batch, 
                  discount,
                  calculate_grad_norm=True):

    rng, *keys_ac = random.split(rng, 4)
    rng, *keys_crt = random.split(rng, 3)
    rng, *keys_cr = random.split(rng, 3)
    
    _, next_actions, next_log_probs, _ = actor.apply_fn(
        {"params": actor.params}, 
        keys_ac,
        batch.next_images, 
        batch.next_proprioceptions,
        True)                           # apply_rad

    critic_target = critic.replace(params=critic_target_params)
    
    target_Q1, target_Q2 = critic_target.apply_fn(
        {"params": critic_target.params}, 
        keys_crt,
        batch.next_images, 
        batch.next_proprioceptions, 
        next_actions,
        True)                           # apply_rad
    
    temp_val = temp.apply_fn({"params": temp.params})
    target_V = jnp.minimum(target_Q1, target_Q2) - temp_val * next_log_probs

    target_Q = batch.rewards + batch.dones * discount * target_V

    def critic_loss_fn(critic_params):
        q1, q2 = critic.apply_fn( 
            {'params': critic_params}, 
            keys_cr,
            batch.images, 
            batch.proprioceptions, 
            batch.actions,
            True)                       # apply_rad
        
        critic_loss = ((q1 - target_Q)**2 + (q2 - target_Q)**2).mean()

        return critic_loss, {
            'critic_loss': critic_loss,
            'q1': q1.mean(),
            'q2': q2.mean()
        }

    grads, info = jax.grad(critic_loss_fn, has_aux=True)(critic.params)
    critic_new = critic.apply_gradients(grads=grads)
    if calculate_grad_norm:
        itr = flax.traverse_util.TraverseTree().iterate(grads)
        info['critic_grad_norm'] = jnp.sqrt(sum([(jnp.linalg.norm(grad)**2) 
                                                 for grad in itr]))
    return rng, critic_new, info


def actor_update(rng, 
                 actor, 
                 critic, 
                 temp, 
                 batch,  
                 calculate_grad_norm=True):
    rng, *keys_ac = random.split(rng, 4)
    rng, *keys_cr = random.split(rng, 3)
    
    temp_val = temp.apply_fn({"params": temp.params})
    
    # The actor's encoder parameters are not updated
    # They are copied from critic's parameters
    if 'encoder' in critic.params:
        actor.params['encoder'] = critic.params['encoder']
        actor = actor.replace(params=actor.params)

    def actor_loss_fn(actor_params):    
        mu, actions, log_probs, _ = actor.apply_fn(
            {"params": actor_params}, 
            keys_ac,
            batch.images, 
            batch.proprioceptions,
            True)                       # apply rad

 
        q1, q2 = critic.apply_fn(
            {'params': critic.params}, 
            keys_cr,
            batch.images, 
            batch.proprioceptions, 
            actions, 
            True)                       # apply rad
        
        q = jnp.minimum(q1, q2)
                
        actor_loss = (log_probs * temp_val - q).mean()

        noise_ratio = jnp.abs(actions - mu).mean()

        return actor_loss, {
            'actor_loss': actor_loss,
            'entropy': -log_probs.mean(),
            'noise_ratio': noise_ratio
        }

    grads, info = jax.grad(actor_loss_fn, has_aux=True)(actor.params)
    actor_new = actor.apply_gradients(grads=grads)

    if calculate_grad_norm:
        itr = flax.traverse_util.TraverseTree().iterate(grads)
        info['actor_grad_norm'] = jnp.sqrt(sum([(jnp.linalg.norm(grad)**2) 
                                                for grad in itr]))

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

def target_update(critic, critic_target_params, tau):
    new_target_params = jax.tree_map(
        lambda p, tp: p * tau + tp * (1 - tau), critic.params,
        critic_target_params)

    return new_target_params