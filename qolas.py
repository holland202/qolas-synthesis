import math,time
from dataclasses import dataclass
from typing import List,Tuple,Dict,Optional
import numpy as np
try:
    import torch,torch.nn as nn,torch.nn.functional as F
except ImportError as e:
    raise ImportError("pip install torch numpy")from e

@dataclass
class QOLASConfig:
    n_qubits:int=4;d:int=16;ptm_size:int=256
    patch_size:int=16;num_patches:int=256
    encoder_embed_dim:int=256
    encoder_num_layers:int=1
    encoder_num_heads:int=4
    encoder_ffn_dim:int=512
    num_layers:int=2
    num_heads:int=4
    num_kv_heads:int=2
    llm_embed_dim:int=128
    ffn_dim:int=256
    projector_hidden_dim:int=256
    batch_size:int=1
    max_seq_length:int=64
    max_synthesis_steps:int=50
    fidelity_threshold:float=0.995
    temperature:float=0.8
    vocab_size:int=259

class PauliBasis:
    def __init__(self,n):
        self.n=n;self.d=2**n
        I,X,Y,Z=np.eye(2,dtype=complex),np.array([[0,1],[1,0]],dtype=complex),np.array([[0,-1j],[1j,0]],dtype=complex),np.array([[1,0],[0,-1]],dtype=complex)
        ops=[np.ones((1,1),dtype=complex)]
        for _ in range(n):
            ops=[np.kron(o,p)for o in ops for p in[I,X,Y,Z]]
        self.operators=ops
    def compute_ptm(self,U):
        d=self.d;P=np.zeros((d*d,d*d))
        UP=np.stack([U@p@U.conj().T for p in self.operators])
        for i,Pi in enumerate(self.operators):
            P[i,:]=np.trace(Pi@UP,axis1=1,axis2=2).real/d
        return P
    def compute_fidelity(self,U1,U2):
        return float(np.abs(np.trace(U1.conj().T@U2))/self.d)

class GateSet:
    PAD,BOS,EOS,UNK=0,1,2,3
    def __init__(self,n):
        self.n=n;self.d=2**n;self.pauli=PauliBasis(n)
        self.token_to_gate={};self.gate_to_token={};self.token_to_name={0:"PAD",1:"BOS",2:"EOS",3:"UNK"}
        self._build()
    def _single(self,M,t):
        I=np.eye(2,dtype=complex);ops=[I]*self.n;ops[t]=M
        r=ops[0]
        for o in ops[1:]:r=np.kron(r,o)
        return r
    def _cnot(self,c,t):
        P0,P1,X,I=np.array([[1,0],[0,0]],dtype=complex),np.array([[0,0],[0,1]],dtype=complex),np.array([[0,1],[1,0]],dtype=complex),np.eye(2,dtype=complex)
        o0=[I]*self.n;o0[c]=P0
        o1=[I]*self.n;o1[c]=P1;o1[t]=X
        U0,U1=o0[0],o1[0]
        for o in o0[1:]:U0=np.kron(U0,o)
        for o in o1[1:]:U1=np.kron(U1,o)
        return U0+U1
    def _cz(self,c,t):
        P0,P1,Z,I=np.array([[1,0],[0,0]],dtype=complex),np.array([[0,0],[0,1]],dtype=complex),np.array([[1,0],[0,-1]],dtype=complex),np.eye(2,dtype=complex)
        o0=[I]*self.n;o0[c]=P0
        o1=[I]*self.n;o1[c]=P1;o1[t]=Z
        U0,U1=o0[0],o1[0]
        for o in o0[1:]:U0=np.kron(U0,o)
        for o in o1[1:]:U1=np.kron(U1,o)
        return U0+U1
    def _swap(self,a,b):
        U=np.zeros((self.d,self.d),dtype=complex)
        for i in range(self.d):
            ba,bb=(i>>a)&1,(i>>b)&1
            j=i^((ba^bb)<<a)^((ba^bb)<<b)
            U[j,i]=1
        return U
    def _add(self,name,M,key):
        tid=len(self.token_to_name);self.token_to_gate[tid]=M;self.gate_to_token[key]=tid;self.token_to_name[tid]=name
    def _build(self):
        S={"H":np.array([[1,1],[1,-1]],dtype=complex)/math.sqrt(2),"X":np.array([[0,1],[1,0]],dtype=complex),"Y":np.array([[0,-1j],[1j,0]],dtype=complex),"Z":np.array([[1,0],[0,-1]],dtype=complex),"S":np.array([[1,0],[0,1j]],dtype=complex),"T":np.array([[1,0],[0,np.exp(1j*math.pi/4)]],dtype=complex),"Sdg":np.array([[1,0],[0,-1j]],dtype=complex),"Tdg":np.array([[1,0],[0,np.exp(-1j*math.pi/4)]],dtype=complex)}
        for n,M in S.items():
            for q in range(self.n):self._add(f"{n}[{q}]",self._single(M,q),(n,q))
        for c in range(self.n):
            for t in range(self.n):
                if c!=t:self._add(f"CNOT[{c},{t}]",self._cnot(c,t),("CNOT",c,t))
        for c in range(self.n):
            for t in range(c+1,self.n):
                self._add(f"CZ[{c},{t}]",self._cz(c,t),("CZ",c,t))
                self._add(f"SWAP[{c},{t}]",self._swap(c,t),("SWAP",c,t))
        self.vocab_size=len(self.token_to_name)
    def apply_sequence(self,seq):
        U=np.eye(self.d,dtype=complex)
        for G in seq:U=G@U
        return U
    def tokens_to_gates(self,toks):return[self.token_to_gate[t]for t in toks if t in self.token_to_gate]
    def random_sequence(self,L):
        v=list(self.token_to_gate.keys());t=[int(np.random.choice(v))for _ in range(L)];g=[self.token_to_gate[x]for x in t]
        return g,t

class RMSNorm(nn.Module):
    def __init__(self,d,eps=1e-6):
        super().__init__();self.eps=eps;self.weight=nn.Parameter(torch.ones(d))
    def forward(self,x):return x*torch.rsqrt(x.pow(2).mean(dim=-1,keepdim=True)+self.eps)*self.weight

class RoPE(nn.Module):
    def __init__(self,hd,max_len=2048,base=10000.0):
        super().__init__();self.hd=hd
        inv=1.0/(base**(torch.arange(0,hd,2).float()/hd))
        t=torch.arange(max_len,dtype=inv.dtype)
        f=torch.outer(t,inv);e=torch.cat([f,f],dim=-1)
        self.register_buffer("cos",e.cos());self.register_buffer("sin",e.sin())
    def forward(self,q,k,T):
        c,s=self.cos[:T].unsqueeze(0).unsqueeze(0),self.sin[:T].unsqueeze(0).unsqueeze(0)
        def rh(x):x1,x2=x[...,::2],x[...,1::2];return torch.stack([-x2,x1],dim=-1).flatten(-2)
        return q*c+rh(q)*s,k*c+rh(k)*s

class GQARoPEAttention(nn.Module):
    def __init__(self,c):
        super().__init__();self.nh=c.num_heads;self.nkv=c.num_kv_heads;self.hd=c.llm_embed_dim//c.num_heads;self.sc=self.hd**-0.5;self.rp=self.nh//self.nkv
        self.qp=nn.Linear(c.llm_embed_dim,self.nh*self.hd,bias=False);self.kp=nn.Linear(c.llm_embed_dim,self.nkv*self.hd,bias=False);self.vp=nn.Linear(c.llm_embed_dim,self.nkv*self.hd,bias=False);self.op=nn.Linear(self.nh*self.hd,c.llm_embed_dim,bias=False);self.rope=RoPE(self.hd)
    def forward(self,x,mask=None):
        B,T,C=x.shape
        q=self.qp(x).view(B,T,self.nh,self.hd).transpose(1,2);k=self.kp(x).view(B,T,self.nkv,self.hd).transpose(1,2);v=self.vp(x).view(B,T,self.nkv,self.hd).transpose(1,2)
        q,k=self.rope(q,k,T)
        if self.rp>1:k=k.repeat_interleave(self.rp,dim=1);v=v.repeat_interleave(self.rp,dim=1)
        a=(q@k.transpose(-2,-1))*self.sc
        if mask is not None:a=a.masked_fill(mask==0,float("-inf"))
        a=F.softmax(a,dim=-1);o=(a@v).transpose(1,2).contiguous().view(B,T,-1)
        return self.op(o)

class SwiGLU(nn.Module):
    def __init__(self,d,h):
        super().__init__();self.w1=nn.Linear(d,h,bias=False);self.w2=nn.Linear(d,h,bias=False);self.w3=nn.Linear(h,d,bias=False)
    def forward(self,x):return self.w3(F.silu(self.w1(x))*self.w2(x))

class QOLASDecoderLayer(nn.Module):
    def __init__(self,c):
        super().__init__();self.attn=GQARoPEAttention(c);self.n1=RMSNorm(c.llm_embed_dim);self.ffn=SwiGLU(c.llm_embed_dim,c.ffn_dim);self.n2=RMSNorm(c.llm_embed_dim)
    def forward(self,x,mask=None):x=x+self.attn(self.n1(x),mask);return x+self.ffn(self.n2(x))

class PTMPatchEncoder(nn.Module):
    def __init__(self,c):
        super().__init__();self.ps=c.patch_size;self.np=c.num_patches;pd=c.patch_size*c.patch_size
        self.pe=nn.Linear(pd,c.encoder_embed_dim);self.pos=nn.Parameter(torch.zeros(1,c.num_patches,c.encoder_embed_dim))
        self.layers=nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=c.encoder_embed_dim,nhead=c.encoder_num_heads,dim_feedforward=c.encoder_ffn_dim,batch_first=True)
            for _ in range(c.encoder_num_layers)
        ])
    def forward(self,ptm):
        B=ptm.size(0);ptm=ptm.unsqueeze(1);p=F.unfold(ptm,kernel_size=self.ps,stride=self.ps).transpose(1,2)
        x=self.pe(p)+self.pos
        for L in self.layers:x=L(x)
        return x

class PTMProjector(nn.Module):
    def __init__(self,c):
        super().__init__();self.mlp=nn.Sequential(nn.Linear(c.encoder_embed_dim,c.projector_hidden_dim),nn.GELU(),nn.Linear(c.projector_hidden_dim,c.llm_embed_dim))
    def forward(self,x):return self.mlp(x)

class QOLASModel(nn.Module):
    def __init__(self,c,v):
        super().__init__();self.config=c;self.pl=c.num_patches;self.vocab_size=v
        self.te=nn.Embedding(v,c.llm_embed_dim);self.enc=PTMPatchEncoder(c);self.proj=PTMProjector(c)
        self.layers=nn.ModuleList([QOLASDecoderLayer(c)for _ in range(c.num_layers)]);self.norm=RMSNorm(c.llm_embed_dim);self.head=nn.Linear(c.llm_embed_dim,v,bias=False)
        self.te.weight=self.head.weight
    def _mask(self,pl,tl,dev):
        m=torch.zeros(tl,tl,device=dev);m[:pl,:pl]=1;m[pl:,:pl]=1;gl=tl-pl
        if gl>0:m[pl:,pl:]=torch.tril(torch.ones(gl,gl,device=dev))
        return m.unsqueeze(0).unsqueeze(0)
    def forward(self,ptm,g):
        B=ptm.size(0);pr=self.proj(self.enc(ptm));ge=self.te(g);x=torch.cat([pr,ge],dim=1)
        m=self._mask(self.pl,x.size(1),x.device)
        for L in self.layers:x=L(x,m)
        return self.head(self.norm(x)[:,self.pl:,:])
    @torch.no_grad()
    def generate(self,ptm,max_steps=50,temp=0.8):
        self.eval();B=ptm.size(0);dev=ptm.device;pr=self.proj(self.enc(ptm));pl=pr.size(1)
        gen=torch.full((B,1),GateSet.BOS,dtype=torch.long,device=dev)
        for _ in range(max_steps):
            ge=self.te(gen);x=torch.cat([pr,ge],dim=1);m=self._mask(pl,x.size(1),dev)
            h=x
            for L in self.layers:h=L(h,m)
            h=self.norm(h);log=self.head(h[:,-1,:])/temp;prb=F.softmax(log,dim=-1);nxt=torch.multinomial(prb,1);gen=torch.cat([gen,nxt],dim=1)
            if(nxt==GateSet.EOS).all():break
        return gen[:,1:]
    def count(self):return sum(p.numel()for p in self.parameters()if p.requires_grad)
    def mem(self):
        b=sum(p.numel()*p.element_size()for p in self.parameters())
        return{"params_mb":b/(1024**2),"params_m":self.count()/1e6}

class SynthesisEngine:
    def __init__(self,m,c,g):self.model=m;self.config=c;self.gs=g
    def synthesize(self,U,strategy="greedy"):
        self.model.eval()
        with torch.no_grad():
            ptm=torch.from_numpy(self.gs.pauli.compute_ptm(U)).float().unsqueeze(0).to(next(self.model.parameters()).device)
            t=1.0 if strategy=="greedy"else self.config.temperature
            tok=self.model.generate(ptm,max_steps=self.config.max_synthesis_steps,temp=t)
        tl=tok[0].cpu().tolist();gs=self.gs.tokens_to_gates(tl);Up=self.gs.apply_sequence(gs);f=self.gs.pauli.compute_fidelity(U,Up)
        return{"gate_sequence":[self.gs.token_to_name.get(t,"?")for t in tl],"tokens":tl,"fidelity":f,"steps":len(gs),"unitary_pred":Up}

class SyntheticDataGenerator:
    def __init__(self,c,g):self.c=c;self.gs=g;self.p=g.pauli
    def generate_batch(self,B,min_len=3,max_len=10):
        P,I,T=[],[],[];ml=self.c.max_seq_length
        for _ in range(B):
            L=np.random.randint(min_len,max_len+1);g,t=self.gs.random_sequence(L);U=self.gs.apply_sequence(g);p=self.p.compute_ptm(U)
            inp=[GateSet.BOS]+t[:-1];tgt=t
            if len(inp)<ml:pad=[GateSet.PAD]*(ml-len(inp));inp+=pad;tgt+=pad
            else:inp=inp[:ml];tgt=tgt[:ml]
            P.append(p);I.append(inp);T.append(tgt)
        return torch.from_numpy(np.stack(P)).float(),torch.tensor(I,dtype=torch.long),torch.tensor(T,dtype=torch.long)

class Trainer:
    def __init__(self,m,c,d):self.model=m;self.config=c;self.dg=d;self.opt=None
    def _sg(self,m,r):
        for p in m.parameters():p.requires_grad=r
    def s1(self,steps=7000,lr=1e-3):
        print(f"\n[Stage 1] Projector+Encoder ({steps} steps, lr={lr})")
        self._sg(self.model.te,False);self._sg(self.model.layers,False);self._sg(self.model.head,False);self._sg(self.model.enc,True);self._sg(self.model.proj,True)
        self.opt=torch.optim.AdamW(list(self.model.enc.parameters())+list(self.model.proj.parameters()),lr=lr)
        self._loop(steps)
    def s2(self,steps=3000,llm_lr=2e-5,proj_lr=8e-5):
        print(f"\n[Stage 2] Joint ({steps} steps, llm={llm_lr}, proj={proj_lr})")
        for p in self.model.parameters():p.requires_grad=True
        self.opt=torch.optim.AdamW([{"params":list(self.model.enc.parameters())+list(self.model.proj.parameters()),"lr":proj_lr},{"params":list(self.model.te.parameters())+list(self.model.layers.parameters())+list(self.model.head.parameters()),"lr":llm_lr}])
        self._loop(steps)
    def _loop(self,steps):
        self.model.train();dev=next(self.model.parameters()).device;t0=time.time()
        for i in range(steps):
            ptm,inp,tgt=self.dg.generate_batch(self.config.batch_size);ptm,inp,tgt=ptm.to(dev),inp.to(dev),tgt.to(dev)
            log=self.model(ptm,inp);loss=F.cross_entropy(log.view(-1,self.model.vocab_size),tgt.view(-1),ignore_index=GateSet.PAD)
            self.opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(self.model.parameters(),1.0);self.opt.step()
            if i%100==0 or i==steps-1:print(f"  Step {i:>5}/{steps} | Loss: {loss.item():.4f} | {time.time()-t0:.1f}s")
        print(f"[Done] Loss: {loss.item():.4f}")

def test():
    print("="*60+"\nQOLAS v1.0 SELF-TEST (MOBILE)\n"+"="*60)
    c=QOLASConfig();g=GateSet(c.n_qubits);c.vocab_size=g.vocab_size;p=g.pauli;ok=0;fail=0
    def check(n,c):
        nonlocal ok,fail
        if c:print(f"  OK {n}");ok+=1
        else:print(f"  FAIL {n}");fail+=1
    print("\n[1/8] PauliBasis");check("256 ops",len(p.operators)==256);check("16x16",p.operators[0].shape==(16,16))
    print("\n[2/8] GateSet");check("vocab>0",g.vocab_size>0);check("CNOT",any("CNOT"in g.token_to_name[k]for k in g.token_to_gate))
    print("\n[3/8] PTM");U=np.eye(16,dtype=complex);P=p.compute_ptm(U);check("256x256",P.shape==(256,256));check("identity",np.allclose(P,np.eye(256),atol=1e-5))
    print("\n[4/8] Fidelity");check("exact=1",abs(p.compute_fidelity(U,U)-1)<1e-6);Ur=np.linalg.qr(np.random.randn(16,16)+1j*np.random.randn(16,16))[0];check("random",0.05<p.compute_fidelity(U,Ur)<0.45)
    print("\n[5/8] Model");m=QOLASModel(c,g.vocab_size);mem=m.mem();check(f"params {mem['params_m']:.1f}M",mem["params_m"]>0);check(f"mem {mem['params_mb']:.1f}MB",mem["params_mb"]<500)
    print("\n[6/8] Forward");log=m(torch.randn(1,256,256),torch.randint(0,g.vocab_size,(1,10)));check(f"shape {tuple(log.shape)}",log.shape==(1,10,g.vocab_size))
    print("\n[7/8] Synthesis");e=SynthesisEngine(m,c,g);r=e.synthesize(U);check("seq","gate_sequence"in r);check("fidelity",isinstance(r["fidelity"],float))
    print("\n[8/8] DataGen");d=SyntheticDataGenerator(c,g);t0=time.time();b=d.generate_batch(50);dt=time.time()-t0;rate=50/dt;check(f"50 in {dt:.1f}s ({rate:.1f}/s)",rate>1);check("shape",b[0].shape==(50,256,256))
    print("\n"+"="*60);print(f"RESULT: {ok}/{ok+fail} PASS");print("="*60);return fail==0

if __name__=="__main__":test()

