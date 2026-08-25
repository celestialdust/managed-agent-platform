"""What a Session's model binding admits, and what it refuses.

Every case here is about a difference, because the failures this module exists to
prevent all return a plausible answer. A slot lookup that fell back to the root, a
provider table missing the one field the runtime does not insist on, a base URL one
segment short -- each renders cleanly and is wrong only where two configurations
differ, so the assertions are written on what one binding says and another does not.
"""

import tomllib
from uuid import uuid4

import pytest

from managed_agent.control.pod_config.model_binding import (
    MODEL_PROVIDER_AUTH_HEADER,
    MODEL_PROVIDER_ID,
    MODEL_PROVIDER_NAME,
    MODEL_PROVIDER_WIRE,
    ROOT_SLOT,
    AgentBinding,
    ModelBindingViolation,
    SessionModelBindings,
    UnboundAgentSlot,
    agent_slot,
    bind_agent,
    bind_session,
    check_agent_document,
    render_agent_document,
    render_model_selection,
)
from managed_agent.core.ids import SessionId, TenantId, new_definition_id
from managed_agent.core.pod.workspace_contract import (
    AGENT_LABEL,
    INPUT_DIR_NAME,
    OUTPUT_DIR_NAME,
    PIP_WRAPPER,
    PLATFORM_LABEL,
    workspace_contract,
)
from managed_agent.core.registration.definition import (
    AgentDefinition,
    MultiAgentPosture,
)
from managed_agent.core.session.session import SessionRecord

GATEWAY_URL = "http://model-gateway.map-dev.svc.cluster.local/v1"
A_MODEL = "gpt-5-codex"


def _definition(model: str = A_MODEL, *, multiagent: bool = False) -> AgentDefinition:
    return AgentDefinition(
        name="slr-extractor",
        instructions="Extract findings and name the source document for each.",
        model=model,
        skills_repository="https://git.internal/skills.git",
        skills_revision="a" * 40,
        multiagent=MultiAgentPosture(enabled=multiagent, max_depth=2),
    )


def _record(revision: str = "3") -> SessionRecord:
    return SessionRecord(
        id=SessionId(uuid4()),
        tenant_id=TenantId(uuid4()),
        definition_id=new_definition_id(),
        definition_revision=revision,
        grant=frozenset(),
        scope=(),
        budget_minor_units=10_000,
        budget_currency="USD",
        retention_days=30,
    )


def _a_binding(slot: str = "slr-appraiser", model: str = A_MODEL) -> AgentBinding:
    return AgentBinding(
        slot=agent_slot(slot),
        model=model,
        instructions="Appraise risk of bias.",
        description="appraiser",
    )


# --------------------------------------------------------------------------------------
# The slot
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["root", "slr-extractor", "a", "a" * 64, "a_b9"])
def test_a_plain_lowercase_name_is_a_slot(name: str) -> None:
    assert agent_slot(name) == name


@pytest.mark.parametrize(
    "name", ["", "Root", "a/b", "../root", "a.b", "a" * 65, "9lives", "-lead", "a b"]
)
def test_a_name_that_would_escape_a_filename_or_a_bare_key_is_refused(
    name: str,
) -> None:
    with pytest.raises(ModelBindingViolation):
        agent_slot(name)


# --------------------------------------------------------------------------------------
# The binding
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("field", ["model", "instructions", "description"])
@pytest.mark.parametrize("value", ["", "   ", "\n"])
def test_a_binding_with_an_empty_field_is_refused(field: str, value: str) -> None:
    fields = {
        "slot": agent_slot("slr-extractor"),
        "model": A_MODEL,
        "instructions": "Extract findings.",
        "description": "extractor",
    }
    fields[field] = value

    with pytest.raises(ModelBindingViolation) as raised:
        AgentBinding(**fields)  # type: ignore[arg-type]

    assert field in str(raised.value)


def test_a_raw_string_that_is_not_a_slot_name_is_refused_at_the_constructor() -> None:
    """NewType is erased at runtime, so the type does not stop this on its own."""
    with pytest.raises(ModelBindingViolation):
        AgentBinding(
            slot="../root",  # type: ignore[arg-type]
            model=A_MODEL,
            instructions="Extract findings.",
            description="extractor",
        )


def test_a_session_with_no_root_binding_is_refused() -> None:
    with pytest.raises(ModelBindingViolation) as raised:
        SessionModelBindings(
            session_id=SessionId(uuid4()),
            definition_revision="1",
            multiagent_enabled=False,
            agents=(_a_binding(),),
        )

    assert ROOT_SLOT in str(raised.value)


def test_a_session_binding_one_slot_twice_is_refused() -> None:
    root = bind_agent(ROOT_SLOT, _definition())

    with pytest.raises(ModelBindingViolation):
        SessionModelBindings(
            session_id=SessionId(uuid4()),
            definition_revision="1",
            multiagent_enabled=False,
            agents=(root, root),
        )


def test_an_unbound_slot_raises_naming_it_rather_than_answering_with_the_root() -> None:
    bindings = bind_session(_record(), _definition())

    with pytest.raises(UnboundAgentSlot) as raised:
        bindings.binding_for(agent_slot("slr-appraiser"))

    assert "slr-appraiser" in str(raised.value)


def test_models_and_permits_cover_every_bound_agent_and_nothing_else() -> None:
    bindings = bind_session(_record(), _definition()).with_agent(
        _a_binding(model="claude-sonnet-4-5-foundry")
    )

    assert bindings.models == frozenset({A_MODEL, "claude-sonnet-4-5-foundry"})
    assert bindings.permits(A_MODEL)
    assert bindings.permits("claude-sonnet-4-5-foundry")
    assert not bindings.permits("gpt-5-codex-mini")


def test_with_agent_returns_a_new_value_and_leaves_the_original_alone() -> None:
    before = bind_session(_record(), _definition())

    after = before.with_agent(_a_binding(model="gpt-5-codex-mini"))

    assert before.agents == (bind_agent(ROOT_SLOT, _definition()),)
    assert len(after.agents) == 2
    assert before.models == frozenset({A_MODEL})


def test_a_second_root_is_refused() -> None:
    bindings = bind_session(_record(), _definition())

    with pytest.raises(ModelBindingViolation):
        bindings.with_agent(bind_agent(ROOT_SLOT, _definition()))


def test_bind_session_carries_the_revision_the_session_pinned() -> None:
    record = _record(revision="7")

    assert bind_session(record, _definition()).definition_revision == "7"


@pytest.mark.parametrize("posture", [True, False])
def test_bind_session_reads_multiagent_off_the_definitions_posture(
    posture: bool,
) -> None:
    bindings = bind_session(_record(), _definition(multiagent=posture))

    assert bindings.multiagent_enabled is posture


def test_a_model_name_no_catalog_knows_is_bound_unchanged() -> None:
    """A model name is a routing key. It fails at the Turn that calls it, not here."""
    bindings = bind_session(_record(), _definition("a-model-nobody-declared"))

    assert bindings.root.model == "a-model-nobody-declared"


# --------------------------------------------------------------------------------------
# The model selection
# --------------------------------------------------------------------------------------


A_TOKEN = "11111111-1111-1111-1111-111111111111.2222.9999999999.abc123"
"""A token shaped like the real one, and deliberately not minted here.

This module is about the document the compiler writes, not about the token's own
signature -- `tests/core/test_session_token.py` owns that. What matters here is that the
value the caller passes reaches the header unchanged and quoted, so a token whose
bytes this file chose is the stronger fixture: a renderer that silently re-minted,
truncated or re-encoded it could not pass.
"""


def _selection(
    model: str = A_MODEL,
    *,
    multiagent: bool = False,
    url: str = GATEWAY_URL,
    token: str = A_TOKEN,
) -> dict[str, object]:
    bindings = bind_session(_record(), _definition(model, multiagent=multiagent))
    return tomllib.loads(
        render_model_selection(bindings, gateway_base_url=url, session_token=token)
    )


def test_the_fragment_names_the_root_model_and_the_one_provider() -> None:
    document = _selection()

    assert document["model"] == A_MODEL
    assert document["model_provider"] == MODEL_PROVIDER_ID


def test_the_provider_entry_is_the_only_one_and_carries_every_field() -> None:
    providers = _selection()["model_providers"]
    assert isinstance(providers, dict)

    assert list(providers) == [MODEL_PROVIDER_ID]
    entry = providers[MODEL_PROVIDER_ID]
    assert entry["name"] == MODEL_PROVIDER_NAME
    assert entry["base_url"] == GATEWAY_URL
    assert entry["wire_api"] == MODEL_PROVIDER_WIRE
    assert entry["http_headers"] == {MODEL_PROVIDER_AUTH_HEADER: f"Bearer {A_TOKEN}"}


def test_the_token_the_caller_passed_is_the_token_the_header_carries() -> None:
    """The one thing between a minted token and a model call that can go wrong here.

    The predecessor of this table named an environment variable instead of carrying
    a value, and nothing filled that variable -- so against real `map-dev` the runtime
    sent the Model Gateway no request at all. What replaced it can fail in exactly one
    way the assertion above would not catch: the value arriving is not the value
    passed. So this drives two different tokens through and requires each back out.
    """
    for token in ("aaaa.bbbb.1.cc", "dddd.eeee.2.ff"):
        entry = _selection(token=token)["model_providers"]
        assert isinstance(entry, dict)
        headers = entry[MODEL_PROVIDER_ID]["http_headers"]
        assert headers == {MODEL_PROVIDER_AUTH_HEADER: f"Bearer {token}"}, (
            f"the provider table carries {headers!r} for a Session token of {token!r}. "
            "The runtime builds its header map once at client construction, so a token "
            "altered here is altered for every model call the pod ever makes."
        )


def test_the_fragment_names_no_variable_for_the_runtime_to_read() -> None:
    """`env_key` and `env_http_headers` both name a variable; nothing fills one.

    `deploy/k8s/session-pod.yaml`'s environment list is exactly four names with no
    `valueFrom` and no `envFrom`, which another slice asserts. A provider table that
    reads a variable therefore reads an unset one, and the runtime's answer to that is
    to send no request -- measured, not inferred. This is the guard that keeps the
    carrier a value rather than a reference, so the failure cannot come back by way of
    a well-meaning edit.
    """
    entry = _selection()["model_providers"]
    assert isinstance(entry, dict)
    provider = entry[MODEL_PROVIDER_ID]
    assert isinstance(provider, dict)

    reads_a_variable = sorted(
        key for key in provider if key in {"env_key", "env_http_headers"}
    )
    assert not reads_a_variable, (
        f"the provider table names {reads_a_variable}, each of which tells the runtime "
        "to read an environment variable. The Session pod's environment is fixed at "
        "four names and nothing adds a fifth, so the variable is unset and the runtime "
        "sends no model request. Carry the token as a value in http_headers instead."
    )


@pytest.mark.parametrize("posture", [True, False])
def test_the_agents_table_states_the_posture_rather_than_inheriting_it(
    posture: bool,
) -> None:
    agents = _selection(multiagent=posture)["agents"]
    assert isinstance(agents, dict)

    assert agents["enabled"] is posture


def test_a_subagent_spawned_with_no_model_lands_on_one_the_session_bound() -> None:
    bindings = bind_session(_record(), _definition())
    document = tomllib.loads(
        render_model_selection(
            bindings, gateway_base_url=GATEWAY_URL, session_token=A_TOKEN
        )
    )

    assert bindings.permits(document["agents"]["default_subagent_model"])


def test_every_bare_key_precedes_every_table_header() -> None:
    """TOML reads a bare key as belonging to the table header above it, so `model`
    written after one would silently become that table's key."""
    text = render_model_selection(
        bind_session(_record(), _definition()),
        gateway_base_url=GATEWAY_URL,
        session_token=A_TOKEN,
    )
    lines = [line for line in text.splitlines() if line and not line.startswith("#")]
    first_table = next(i for i, line in enumerate(lines) if line.startswith("["))

    assert all("=" in line for line in lines[:first_table])
    assert lines[:first_table]


def test_the_fragment_names_no_second_provider_and_no_first_party_base_url() -> None:
    text = render_model_selection(
        bind_session(_record(), _definition()),
        gateway_base_url=GATEWAY_URL,
        session_token=A_TOKEN,
    )

    assert "openai_base_url" not in text
    assert "chatgpt_base_url" not in text
    assert text.count("[model_providers.") == 1


def test_the_provider_id_reads_as_neither_first_party_openai_nor_azure() -> None:
    """The runtime decides from a provider's name and base URL whether it is talking to
    OpenAI or to Azure, and turns on features this service does not serve if it is."""
    assert "openai" not in MODEL_PROVIDER_ID.lower()
    assert "azure" not in MODEL_PROVIDER_ID.lower()
    assert "openai" not in MODEL_PROVIDER_NAME.lower()
    assert "azure" not in MODEL_PROVIDER_NAME.lower()


@pytest.mark.parametrize(
    "url",
    [
        "model-gateway.map-dev.svc.cluster.local/v1",
        "ftp://model-gateway.map-dev.svc.cluster.local/v1",
        "http://model-gateway.map-dev.svc.cluster.local/",
        "http://model-gateway.map-dev.svc.cluster.local",
        "http://model-gateway.map-dev.svc.cluster.local/v2",
        "http://model-gateway.map-dev.svc.cluster.local/v1/responses",
    ],
)
def test_a_base_url_that_would_miss_the_responses_path_is_refused(url: str) -> None:
    bindings = bind_session(_record(), _definition())

    with pytest.raises(ModelBindingViolation) as raised:
        render_model_selection(bindings, gateway_base_url=url, session_token=A_TOKEN)

    assert url in str(raised.value)


@pytest.mark.parametrize(
    "url",
    [
        "http://model-gateway.map-dev.svc.cluster.local/v1",
        "http://model-gateway.map-dev.svc.cluster.local/v1/",
        "https://model-gateway.map.internal/v1",
    ],
)
def test_a_base_url_the_runtime_would_reach_v1_responses_on_is_accepted(
    url: str,
) -> None:
    """The runtime joins base and path as `{base.rstrip('/')}/responses`, so a trailing
    slash is not a difference and demanding one would refuse a working address."""
    providers = _selection(url=url)["model_providers"]
    assert isinstance(providers, dict)

    assert providers[MODEL_PROVIDER_ID]["base_url"] == url


@pytest.mark.parametrize("model", ['a"quoted"model', "a\\backslash\\model", "a\tmodel"])
def test_a_model_name_carrying_toml_syntax_round_trips_unchanged(model: str) -> None:
    assert _selection(model)["model"] == model


def test_a_model_name_carrying_a_control_character_is_refused() -> None:
    with pytest.raises(ModelBindingViolation):
        _selection("a\x00model")


# --------------------------------------------------------------------------------------
# The agent document
# --------------------------------------------------------------------------------------


def test_a_rendered_document_carries_the_bindings_own_configuration() -> None:
    binding = _a_binding(model="claude-sonnet-4-5-foundry")

    document = tomllib.loads(render_agent_document(binding).toml)

    assert document["name"] == binding.slot
    assert document["model"] == "claude-sonnet-4-5-foundry"
    assert document["developer_instructions"] == binding.instructions
    assert document["description"] == binding.description


def test_a_rendered_documents_key_set_is_exactly_the_permitted_one() -> None:
    document = tomllib.loads(render_agent_document(_a_binding()).toml)

    assert set(document) == {"name", "description", "model", "developer_instructions"}


@pytest.mark.parametrize(
    "surplus",
    [
        'sandbox_mode = "danger-full-access"',
        "mcp_servers = []",
        'cwd = "/tmp"',
        "[skills.config]\nenabled = true",
        'approval_policy = "never"',
    ],
)
def test_a_document_that_would_widen_anything_is_refused(surplus: str) -> None:
    document = render_agent_document(_a_binding()).toml

    check_agent_document(document)
    with pytest.raises(ModelBindingViolation) as raised:
        check_agent_document(f"{document}\n{surplus}\n")

    assert "override" in str(raised.value)


def test_a_document_missing_a_permitted_key_is_refused() -> None:
    with pytest.raises(ModelBindingViolation) as raised:
        check_agent_document('name = "a"\nmodel = "b"\ndescription = "c"\n')

    assert "developer_instructions" in str(raised.value)


def test_instructions_spanning_lines_and_carrying_toml_syntax_round_trip() -> None:
    instructions = 'Extract "effect size".\nQuote it.\nEscape a \\ if you see one.'
    binding = AgentBinding(
        slot=agent_slot("slr-extractor"),
        model=A_MODEL,
        instructions=instructions,
        description="extractor",
    )

    document = tomllib.loads(render_agent_document(binding).toml)

    assert document["developer_instructions"] == instructions


def test_the_root_has_no_agent_document() -> None:
    with pytest.raises(ModelBindingViolation):
        render_agent_document(bind_agent(ROOT_SLOT, _definition()))


def test_the_filename_is_the_slot_and_holds_no_path_separator() -> None:
    document = render_agent_document(_a_binding("slr-appraiser"))

    assert document.filename == "slr-appraiser.toml"
    assert "/" not in document.filename


# --------------------------------------------------------------------------------------
# The tenant's own instructions
# --------------------------------------------------------------------------------------


def test_the_definitions_instructions_reach_the_document_the_root_agent_reads() -> None:
    """**The case for a defect that shipped and cost nothing to detect.**

    A definition's `instructions` were bound onto `AgentBinding` and the only reader was
    `render_agent_document`, which refuses the root slot -- so a subagent got its
    instructions and the root agent, which is the thread every Session wraps, got none.
    Measured on a live pod: a definition saying "your codeword is X", asked for its
    codeword, answered `NO CODEWORD`. A tenant's whole persona was discarded in silence.
    """
    definition = _definition()
    text = render_model_selection(
        bind_session(_record(), definition),
        gateway_base_url=GATEWAY_URL,
        session_token=A_TOKEN,
    )
    parsed = tomllib.loads(text)

    # `in` and not `==`: this field carries the platform's workspace contract too, for
    # the reason `render_model_selection` gives -- the runtime key that would keep them
    # apart is in no released codex. What must hold is that the tenant's text arrives
    # whole, which a substring check over the round-tripped value establishes and a
    # regex over the rendered TOML would not.
    assert definition.instructions in parsed["developer_instructions"]


def test_both_authors_of_the_instructions_field_are_labelled() -> None:
    """One field, two authors, and nothing but these labels separates them.

    The field exists to carry the tenant's text. The platform's workspace contract is
    in there as well because the runtime's administrator channel,
    `additional_developer_instructions`, appears zero times in both released codex
    binaries -- so a reader of a Session's compiled `config.toml`, and the model
    itself, has only the labels to tell whose sentence is whose.

    Asserted on the composed field rather than on `instructions_for_the_model`, which
    has its own cases: what would break a tenant is this renderer dropping the labels
    while still concatenating, and that is a defect in this function.
    """
    definition = _definition()
    carried = tomllib.loads(
        render_model_selection(
            bind_session(_record(), definition),
            gateway_base_url=GATEWAY_URL,
            session_token=A_TOKEN,
        )
    )["developer_instructions"]

    assert f"<{PLATFORM_LABEL}>" in carried
    assert f"</{PLATFORM_LABEL}>" in carried
    assert f"<{AGENT_LABEL}>" in carried
    assert f"</{AGENT_LABEL}>" in carried
    # The tenant's text inside the agent block and not the platform's, which is the
    # half a reader would trust as coming from whoever runs this platform.
    platform_half = carried.split(f"</{PLATFORM_LABEL}>")[0]
    assert definition.instructions not in platform_half
    assert workspace_contract() in platform_half


def test_the_platform_half_names_the_directory_the_shim_actually_ships() -> None:
    """The clause that made this field worth changing.

    A Turn that rendered a PDF returned the PDF and the generator script the agent
    wrote to make it, because ship-out took every file at the workspace root. The fix
    is a directory the agent has to be told about, so this asserts the telling reaches
    the document -- an untold convention is the same defect with more code behind it.
    """
    carried = tomllib.loads(
        render_model_selection(
            bind_session(_record(), _definition()),
            gateway_base_url=GATEWAY_URL,
            session_token=A_TOKEN,
        )
    )["developer_instructions"]

    assert f"./{OUTPUT_DIR_NAME}/" in carried
    assert f"./{INPUT_DIR_NAME}/" in carried
    assert PIP_WRAPPER in carried


def test_the_instructions_land_at_top_level_and_not_inside_a_table() -> None:
    """The nesting is the failure mode with no other symptom.

    `developer_instructions` one line lower parses as `agents.developer_instructions`, a
    key the runtime does not read, in a document that loads without complaint -- so the
    tenant's text is in the file, visible to anyone who opens it, and invisible to the
    model. `test_every_bare_key_precedes_every_table_header` above covers the general
    rule; this asserts the specific key, because that rule holding today does not stop
    somebody appending this one to the end of the function tomorrow.
    """
    text = render_model_selection(
        bind_session(_record(), _definition()),
        gateway_base_url=GATEWAY_URL,
        session_token=A_TOKEN,
    )

    assert text.index("developer_instructions") < text.index("[agents]")
    assert "developer_instructions" not in tomllib.loads(text)["agents"]


def test_the_document_never_sets_the_runtimes_own_base_instructions() -> None:
    """`base_instructions` is the runtime's built-in system prompt, not a persona slot.

    It is the text teaching the model `apply_patch`, the shell conventions and its own
    tool protocol. Writing a tenant's paragraph there would not add instructions, it
    would delete the runtime's operating manual -- leaving a model that cannot edit a
    file, for a reason nothing in this repository would explain. `thread/start` accepts
    both fields, which is exactly why this asserts we only ever send the additive one.
    """
    text = render_model_selection(
        bind_session(_record(), _definition()),
        gateway_base_url=GATEWAY_URL,
        session_token=A_TOKEN,
    )

    assert "base_instructions" not in text


def test_instructions_that_run_to_many_lines_survive_the_document() -> None:
    """A raw newline in a TOML basic string is a parse error, not a line break.

    Round-tripped rather than pattern-matched: the assertion is that what the tenant
    wrote is what `tomllib` reads back, which is the only property that matters and the
    one a regex over the rendered text cannot establish.
    """
    definition = _definition().model_copy(
        update={
            "instructions": (
                "You are a careful analyst.\n"
                "\n"
                "Rules:\n"
                '  - Quote sources verbatim, inside "double quotes".\n'
                "  - Refuse to guess; say what you could not determine.\n"
                "\tIndented with a real tab.\n"
            )
        }
    )
    text = render_model_selection(
        bind_session(_record(), definition),
        gateway_base_url=GATEWAY_URL,
        session_token=A_TOKEN,
    )

    assert definition.instructions in tomllib.loads(text)["developer_instructions"]
